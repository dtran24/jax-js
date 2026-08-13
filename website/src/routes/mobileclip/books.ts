import { cachedFetch } from "@jax-js/loaders";

/** A book in text format, split up into excerpts for individual embedding. */
export interface Book {
  title: string;
  author: string;
  chapters: Chapter[];
}

/**
 * A chapter of the book with title and excerpts.
 *
 * Each excerpt is at least a sentence but may be more, if the sentences are
 * short, based on a heuristic to get interesting chunks of text.
 *
 * Some text may be excluded if it's too short, e.g., dialog.
 */
export interface Chapter {
  title: string;
  excerpts: string[];
}

export async function downloadBook(id: string): Promise<Book> {
  switch (id) {
    case "dickens-great-expectations":
      return await dickensGreatExpectations();
    case "wilde-dorian-gray":
      return await wildeDorianGray();
    case "fitzgerald-great-gatsby":
      return await fitzgeraldGreatGatsby();
    default:
      throw new Error(`Unknown book ID: ${id}`);
  }
}

/** Build a single-chapter book from raw uploaded text. */
export function bookFromText(title: string, text: string): Book {
  return {
    title,
    author: "",
    chapters: [{ title: "Full text", excerpts: splitExcerpts(text) }],
  };
}

/** Split text into excerpts by paragraphs, with size constraints. */
function splitExcerpts(text: string): string[] {
  const MIN_LENGTH = 40;
  const MAX_LENGTH = 280;

  const paragraphs = text
    .split(/\n\n+/)
    .map((p) => p.replace(/\n/g, " ").trim());
  const excerpts: string[] = [];

  for (const para of paragraphs) {
    // Skip short paragraphs or horizontal rules
    if (para.length < MIN_LENGTH || /^-+$/.test(para)) {
      continue;
    }

    if (para.length <= MAX_LENGTH) {
      excerpts.push(para);
    } else {
      // Split long paragraphs by sentences (include trailing quotes)
      const sentences = para.match(
        /[^.!?]+[.!?]+["'\u201c\u201d\u2018\u2019]?/g,
      ) || [para];
      let current = "";

      for (const sentence of sentences) {
        const trimmed = sentence.trim();
        if (!current) {
          // Always include at least one sentence, even if it exceeds MAX_LENGTH
          current = trimmed;
        } else if (current.length + trimmed.length + 1 <= MAX_LENGTH) {
          current = current + " " + trimmed;
        } else {
          if (current.length >= MIN_LENGTH) {
            excerpts.push(current);
          }
          current = trimmed;
        }
      }

      if (current.length >= MIN_LENGTH) {
        excerpts.push(current);
      }
    }
  }

  return excerpts;
}

async function dickensGreatExpectations(): Promise<Book> {
  // https://www.gutenberg.org/ebooks/1400
  const url =
    "https://huggingface.co/datasets/ekzhang/jax-js-examples/raw/main/pg1400.txt";

  const dataBytes = await cachedFetch(url);
  let content = new TextDecoder().decode(dataBytes).replace(/\r/g, "");

  // Strip Project Gutenberg license footer
  const endMarker = content.indexOf("*** END OF THE PROJECT GUTENBERG");
  if (endMarker !== -1) {
    content = content.slice(0, endMarker);
  }

  // Split by chapter headings (Chapter I., Chapter II., etc.)
  const chapterRegex =
    /(?:^|\n)(?:\[Illustration\]\s*)?Chapter ([IVXLC]+)\.\s*\n/g;
  const chapters: Chapter[] = [];

  const matches = [...content.matchAll(chapterRegex)];

  for (let i = 0; i < matches.length; i++) {
    const match = matches[i];
    const startIndex = match.index! + match[0].length;
    const endIndex =
      i < matches.length - 1 ? matches[i + 1].index! : content.length;

    const chapterText = content.slice(startIndex, endIndex).trim();
    const excerpts = splitExcerpts(chapterText);

    chapters.push({
      title: `Chapter ${match[1]}`,
      excerpts,
    });
  }

  return {
    title: "Great Expectations",
    author: "Charles Dickens",
    chapters,
  };
}

async function wildeDorianGray(): Promise<Book> {
  // https://www.gutenberg.org/ebooks/4078
  const url =
    "https://huggingface.co/datasets/ekzhang/jax-js-examples/raw/main/pg4078.txt";

  const dataBytes = await cachedFetch(url);
  let content = new TextDecoder().decode(dataBytes).replace(/\r/g, "");

  // Strip Project Gutenberg license footer
  const endMarker = content.indexOf("*** END OF THE PROJECT GUTENBERG");
  if (endMarker !== -1) {
    content = content.slice(0, endMarker);
  }

  // Split by chapter headings (CHAPTER I, CHAPTER II, etc.)
  const chapterRegex = /\nCHAPTER ([IVXLC]+)\n/g;
  const chapters: Chapter[] = [];

  const matches = [...content.matchAll(chapterRegex)];

  for (let i = 0; i < matches.length; i++) {
    const match = matches[i];
    const startIndex = match.index! + match[0].length;
    const endIndex =
      i < matches.length - 1 ? matches[i + 1].index! : content.length;

    const chapterText = content.slice(startIndex, endIndex).trim();
    const excerpts = splitExcerpts(chapterText);

    chapters.push({
      title: `Chapter ${match[1]}`,
      excerpts,
    });
  }

  return {
    title: "The Picture of Dorian Gray",
    author: "Oscar Wilde",
    chapters,
  };
}

async function fitzgeraldGreatGatsby(): Promise<Book> {
  // https://www.gutenberg.org/ebooks/64317
  const url =
    "https://huggingface.co/datasets/ekzhang/jax-js-examples/raw/main/pg64317.txt";

  const dataBytes = await cachedFetch(url);
  let content = new TextDecoder().decode(dataBytes).replace(/\r/g, "");

  // Strip Project Gutenberg license footer
  const endMarker = content.indexOf("*** END OF THE PROJECT GUTENBERG");
  if (endMarker !== -1) {
    content = content.slice(0, endMarker);
  }

  // Split by chapter headings (I, II, III, etc. centered with spaces)
  const chapterRegex = /\n +([IVXLC]+)\s*\n/g;
  const chapters: Chapter[] = [];

  const matches = [...content.matchAll(chapterRegex)];

  for (let i = 0; i < matches.length; i++) {
    const match = matches[i];
    const startIndex = match.index! + match[0].length;
    const endIndex =
      i < matches.length - 1 ? matches[i + 1].index! : content.length;

    const chapterText = content.slice(startIndex, endIndex).trim();
    const excerpts = splitExcerpts(chapterText);

    chapters.push({
      title: `Chapter ${match[1]}`,
      excerpts,
    });
  }

  return {
    title: "The Great Gatsby",
    author: "F. Scott Fitzgerald",
    chapters,
  };
}
