<script lang="ts">
  import {
    defaultDevice,
    init,
    jit,
    numpy as np,
    tree,
    vmap,
  } from "@jax-js/jax";
  import { opfs, safetensors, tokenizers } from "@jax-js/loaders";
  import { BookMarkedIcon, FileTextIcon } from "@lucide/svelte";

  import DownloadManager from "$lib/common/DownloadManager.svelte";
  import { type Book, bookFromText, downloadBook } from "./books";
  import {
    fromSafetensors,
    type MobileCLIP,
    runMobileCLIPTextEncoder,
  } from "./clipInference";

  // Cached large objects to download.
  let _weights: safetensors.File | null = null;
  let _model: MobileCLIP | null = null;
  let _tokenizer: tokenizers.BpeEncoding | null = null;

  // Rough estimate for FLOPs per text embed.
  // - 38e6 ~= number of non-embedding parameters in text encoder
  // - 77 = clip context length
  const GFLOP_PER_TEXT_EMBED = (2 * 38e6 * 77) / 1e9;
  const D_EMBED = 512;

  let downloadManager: DownloadManager;
  let fileInput: HTMLInputElement | undefined = $state();

  async function downloadClipWeights(): Promise<safetensors.File> {
    if (_weights) return _weights;
    isDownloadingWeights = true;
    try {
      const weightsUrl =
        "https://huggingface.co/ekzhang/jax-js-models/resolve/main/mobileclip2-s0.safetensors";

      const data = await downloadManager.fetch("model weights", weightsUrl);
      const result = safetensors.parse(data);
      _weights = result;
      return result;
    } catch (error) {
      alert("Error downloading weights: " + error);
      throw error;
    } finally {
      isDownloadingWeights = false;
    }
  }

  async function getModel(): Promise<MobileCLIP> {
    if (_model) return _model;
    const weights = await downloadClipWeights();
    _model = fromSafetensors(weights);
    hasModel = true;
    return _model;
  }

  async function getTokenizer() {
    if (!_tokenizer) _tokenizer = await tokenizers.getBpe("clip");
    return _tokenizer;
  }

  let hasModel = $state(false);
  let isDownloadingWeights = $state(false);

  let hasData = $state(false);
  let isDownloadingData = $state(false);

  let book = $state<Book>(null as any);
  let embeddingProgress = $state<number[]>([]);
  let embeddingTotal = $state<number>(0);
  let embeddingGflops = $state<number>(0);
  let embeddingArray: np.Array;

  // Flat list mapping excerpt index to { chapterIdx, excerptIdx, text }
  let excerptList = $state<
    { chapterIdx: number; excerptIdx: number; text: string }[]
  >([]);

  // Search state
  let searchQuery = $state("");
  let searchResults = $state<
    { chapterIdx: number; excerptIdx: number; text: string; score: number }[]
  >([]);
  let isSearching = $state(false);

  const numExcerpts = $derived(
    book
      ? book.chapters.map((c) => c.excerpts.length).reduce((a, b) => a + b, 0)
      : 0,
  );

  const runEncoder = jit(vmap(runMobileCLIPTextEncoder, [null, 0]));

  async function setupBook(bookId: string) {
    await runSetup(() => downloadBook(bookId));
  }

  async function setupBookFromFile(file: File) {
    await runSetup(async () => {
      const text = await file.text();
      const title = file.name.replace(/\.[^./]+$/, "");
      const uploaded = bookFromText(title, text);
      if (uploaded.chapters[0].excerpts.length === 0) {
        throw new Error("No usable text found in file");
      }
      return uploaded;
    });
  }

  async function handleFileChange(event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    input.value = "";
    if (file) await setupBookFromFile(file);
  }

  function triggerFileUpload() {
    fileInput?.click();
  }

  async function runSetup(loadBook: () => Promise<Book>) {
    const devices = await init("webgpu");
    if (!devices.includes("webgpu")) {
      alert(
        "WebGPU is not enabled on this browser, try on Chrome or upgrade to iOS 26.",
      );
      return;
    }
    defaultDevice("webgpu");

    const model = await getModel();
    const tokenizer = await getTokenizer();

    isDownloadingData = true;
    try {
      book = await loadBook();
    } catch (error: any) {
      alert("Error loading book: " + error.message);
      return;
    } finally {
      isDownloadingData = false;
    }
    console.log($state.snapshot(book));
    hasData = true;

    const numExcerpts = book.chapters
      .map((c) => c.excerpts.length)
      .reduce((a, b) => a + b, 0);
    console.log(`Total excerpts: ${numExcerpts}`);

    // Tokenization and build excerpt list
    const startTime = performance.now();
    const tokens: number[][] = [];
    const excerptToChapter: number[] = []; // Maps excerpt index to chapter index
    excerptList = [];
    for (let ci = 0; ci < book.chapters.length; ci++) {
      for (let ei = 0; ei < book.chapters[ci].excerpts.length; ei++) {
        const excerpt = book.chapters[ci].excerpts[ei];
        tokens.push(tokenizer.encode(excerpt));
        excerptToChapter.push(ci);
        excerptList.push({ chapterIdx: ci, excerptIdx: ei, text: excerpt });
      }
    }
    const endTime = performance.now();
    console.log(
      `Tokenized ${numExcerpts} excerpts in ${endTime - startTime} ms`,
    );

    const ar = np.array(tokens, { dtype: np.uint32 });

    // Initialize progress tracking
    embeddingProgress = new Array(book.chapters.length).fill(0);
    embeddingTotal = 0;

    // We'll append to this array as we compute embeddings.
    embeddingArray = np.zeros([0, D_EMBED], { dtype: np.float16 });

    try {
      console.log(
        "total params:",
        tree
          .flatten(model.text)[0]
          .map((x) => x.size)
          .reduce((a, b) => a + b, 0),
      );

      for (let i = 0; i < ar.shape[0]; i += 16) {
        const batch = ar.ref.slice([i, Math.min(i + 16, ar.shape[0])]);
        const batchSize = batch.shape[0];
        performance.mark("clip-start");
        const t0 = performance.now();
        const result = runEncoder(tree.ref(model.text), batch);
        await result.blockUntilReady();
        const t1 = performance.now();
        performance.mark("clip-end");
        performance.measure("clip", "clip-start", "clip-end");

        embeddingArray = np.concatenate([embeddingArray, result], 0);
        await embeddingArray.blockUntilReady();

        // Update progress for each excerpt in this batch
        for (let j = i; j < i + batchSize; j++) {
          const chapterIdx = excerptToChapter[j];
          embeddingProgress[chapterIdx]++;
        }
        embeddingProgress = embeddingProgress; // Trigger reactivity
        embeddingTotal += batchSize;

        const gflopsPerSec =
          (GFLOP_PER_TEXT_EMBED * batchSize) / (1e-3 * (t1 - t0));
        embeddingGflops = gflopsPerSec;
        console.log(
          `Processed rows ${i} to ${i + batchSize} in ${t1 - t0} ms (${gflopsPerSec} GFLOP/s)`,
        );
      }
      ar.dispose();
    } catch (error) {
      console.error("Error in main:", error);
    }
  }

  let pendingQuery: string | null = null;
  let searchInProgress = false;

  async function search(query: string) {
    // If a search is in progress, queue this query for later
    if (searchInProgress) {
      pendingQuery = query;
      return;
    }

    if (!query.trim() || !embeddingArray || embeddingTotal === 0) {
      searchResults = [];
      return;
    }

    searchInProgress = true;
    isSearching = true;
    try {
      const model = await getModel();
      const tokenizer = await getTokenizer();

      // Tokenize the query
      const queryTokens = tokenizer.encode(query);
      const queryArray = np.array([queryTokens], { dtype: np.uint32 });

      // Run the encoder to get embeddings (~100 ms?)
      const queryEmbed = runEncoder(tree.ref(model.text), queryArray).slice(0);

      // Compute cosine similarity scores: query @ embeddings.T
      // queryEmbed is [D_EMBED], embeddingArray is [N, D_EMBED]
      const scores: number[] = await np
        .dot(
          embeddingArray.ref.astype(np.float32),
          queryEmbed.astype(np.float32),
        )
        .jsAsync();

      // Argsort descending
      const indices = Array.from({ length: scores.length }, (_, i) => i);
      indices.sort((a, b) => scores[b] - scores[a]);

      // Get top 10 results
      const topK = 10;
      searchResults = indices.slice(0, topK).map((idx) => ({
        ...excerptList[idx],
        score: scores[idx],
      }));
    } catch (error) {
      console.error("Search error:", error);
    } finally {
      searchInProgress = false;
      isSearching = false;

      // If there's a pending query, run it now
      if (pendingQuery !== null) {
        const nextQuery = pendingQuery;
        pendingQuery = null;
        search(nextQuery);
      }
    }
  }

  async function clearCache() {
    try {
      await opfs.clear();
      console.log("Cache cleared");
    } catch (error) {
      console.error("Error clearing cache:", error);
    }
  }
</script>

<DownloadManager bind:this={downloadManager} />

<div class="min-h-screen bg-white">
  <!-- Header with search bar -->

  {#if false}
    <header class="border-b border-gray-200">
      <div class="max-w-4xl mx-auto px-4 py-8">
        <div class="flex items-center gap-4 mb-4">
          <button
            onclick={getModel}
            disabled={isDownloadingWeights || hasModel}
            class="btn"
          >
            {isDownloadingWeights
              ? "Loading…"
              : hasModel
                ? "Model downloaded ✔️"
                : "Download model"}
          </button>
          <button
            onclick={clearCache}
            class="px-4 py-2 border-2 border-black hover:bg-black hover:text-white transition-colors"
          >
            Clear Cache
          </button>
        </div>
      </div>
    </header>
  {/if}

  <!-- Main content area -->
  <main class="max-w-4xl mx-auto px-4 py-8">
    <!-- Empty state -->
    {#if !hasData}
      <div class="text-center py-16">
        <h2 class="text-3xl font-normal mb-2">No data yet</h2>
        <p class="text-lg text-gray-600 mb-12">
          Download and embed a dataset to get started.
        </p>

        <div class="flex flex-col items-center gap-8">
          <div class="w-full max-w-md">
            <h3 class="font-medium mb-4">Load a prepared dataset</h3>
            <div class="flex flex-col gap-3">
              <button
                class="btn"
                onclick={() => setupBook("dickens-great-expectations")}
                disabled={isDownloadingWeights || isDownloadingData}
              >
                <BookMarkedIcon size={20} />
                <span><em>Great Expectations</em> by Charles Dickens</span>
              </button>
              <button
                class="btn"
                onclick={() => setupBook("wilde-dorian-gray")}
                disabled={isDownloadingWeights || isDownloadingData}
              >
                <BookMarkedIcon size={20} />
                <span><em>The Picture of Dorian Gray</em> by Oscar Wilde</span>
              </button>
              <button
                class="btn"
                onclick={() => setupBook("fitzgerald-great-gatsby")}
                disabled={isDownloadingWeights || isDownloadingData}
              >
                <BookMarkedIcon size={20} />
                <span><em>The Great Gatsby</em> by F. Scott Fitzgerald</span>
              </button>
            </div>
          </div>

          <div class="text-sm text-gray-400 uppercase tracking-wider">or</div>

          <div class="w-full max-w-md">
            <h3 class="font-medium mb-4">Upload your own data</h3>
            <div class="flex flex-col gap-3">
              <input
                type="file"
                accept=".txt,text/plain"
                bind:this={fileInput}
                onchange={handleFileChange}
                class="hidden"
              />
              <button
                class="btn"
                onclick={triggerFileUpload}
                disabled={isDownloadingWeights || isDownloadingData}
              >
                <FileTextIcon size={20} />
                Upload a text file (.txt)
              </button>
              <p class="text-xs text-gray-400">
                The parser works best with plain-prose books like the ones
                above. Try a
                <a
                  href="https://www.gutenberg.org/ebooks/search/?sort_order=downloads"
                  target="_blank"
                  rel="noopener noreferrer"
                  class="text-gray-600 hover:text-primary underline"
                >
                  Project Gutenberg
                </a>
                plain text download.
              </p>
            </div>
          </div>
        </div>
      </div>
    {:else}
      <section class="mb-8">
        <form
          class="mb-4"
          onsubmit={(e) => {
            e.preventDefault();
            search(searchQuery);
          }}
        >
          <input
            type="text"
            placeholder="Search excerpts…"
            class="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:border-primary focus:outline-none"
            bind:value={searchQuery}
            oninput={() => search(searchQuery)}
            disabled={embeddingTotal === 0}
          />
        </form>

        {#if isSearching}
          <p class="text-gray-500">Searching…</p>
        {:else if searchResults.length > 0}
          <div
            class="grid grid-cols-[auto_1fr_auto] gap-x-4 gap-y-2 items-baseline"
          >
            {#each searchResults as result, i (i)}
              <span class="text-sm font-medium text-primary whitespace-nowrap">
                {book.chapters[result.chapterIdx].title}
              </span>
              <p class="text-sm text-gray-700">{result.text}</p>
              <span class="text-xs text-gray-400"
                >{result.score.toFixed(3)}</span
              >
            {/each}
          </div>
        {:else if searchQuery.trim() && embeddingTotal > 0}
          <p class="text-gray-500">No results found.</p>
        {:else if embeddingTotal === 0}
          <p class="text-gray-500">Waiting for embeddings to complete…</p>
        {/if}
      </section>

      <section class="border-primary border-2 bg-primary/10 rounded-xl p-6">
        <div class="mb-4">
          <h2 class="text-lg font-semibold mb-0.5">
            Embedding: <em>{book.title}</em>
            {#if embeddingTotal === numExcerpts}
              ✅
            {/if}
          </h2>
          <p class="text-gray-600 text-sm">
            {embeddingTotal < numExcerpts
              ? "Currently embedding"
              : "Generated embeddings for"}
            {numExcerpts.toLocaleString()} excerpts with MobileCLIP2.
            {#if embeddingGflops}
              <span class="font-bold">{embeddingGflops.toFixed(2)} GFLOP/s</span
              >
            {/if}
          </p>
        </div>
        <div class="grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 items-center">
          {#each book.chapters as chapter, i (i)}
            <p class="max-w-[24ch] truncate text-sm">{chapter.title}</p>
            <div
              class="w-full bg-white/75 border border-primary/20 rounded-full h-4 overflow-hidden"
            >
              <div
                class="bg-primary h-3.5 transition-all duration-150 ease-linear"
                style="width: {chapter.excerpts.length > 0
                  ? (100 * (embeddingProgress[i] || 0)) /
                    chapter.excerpts.length
                  : 0}%"
              ></div>
            </div>
          {/each}
        </div>
      </section>
    {/if}
  </main>
</div>

<style lang="postcss">
  @reference "$app.css";

  .btn {
    @apply flex items-center justify-center gap-2 px-5 py-2.5 border-2 border-black;
    @apply enabled:hover:bg-black enabled:hover:text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors;
  }
</style>
