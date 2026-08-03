import {
  AluExp,
  AluOp,
  byteWidth,
  DType,
  dtypedArray,
  isFloatDtype,
  Kernel,
} from "../alu";
import { Backend, Device, Executable, Slot, SlotError } from "../backend";
import { Routine, runCpuRoutine } from "../routine";
import { tuneNullopt } from "../tuner";

/**
 * Rewrite `Signbit` of a float global load into an integer load of the
 * element's sign byte, via a raw byte view of the same buffer exposed as
 * global `numInputs + gid`.
 *
 * JS engines may canonicalize NaN when reading a float out of a typed array,
 * losing its sign, so the sign bit must be read as an integer instead.
 */
function rewriteSignbitLoads(exp: AluExp, numInputs: number): AluExp {
  return exp.rewrite((e) => {
    if (e.op !== AluOp.Signbit) return;
    const input = e.src[0];
    if (input.op !== AluOp.GlobalIndex || !isFloatDtype(input.dtype)) return;
    const [gid, len] = input.arg as [number, number];
    const width = byteWidth(input.dtype);
    // Little-endian: the sign bit is in the last byte of the element.
    const byteIdx = AluExp.add(
      AluExp.mul(input.src[0], AluExp.i32(width)),
      AluExp.i32(width - 1),
    );
    const signByte = AluExp.globalIndex(
      DType.Int32,
      numInputs + gid,
      len * width,
      byteIdx,
    );
    return AluExp.cmplt(AluExp.i32(127), signByte);
  });
}

/** Most basic implementation of `Backend` for testing. */
export class CpuBackend implements Backend {
  readonly type: Device = "cpu";
  readonly maxArgs = Infinity;

  #buffers: Map<Slot, { ref: number; buffer: Uint8Array<ArrayBuffer> }>;
  #nextSlot: number;

  constructor() {
    this.#buffers = new Map();
    this.#nextSlot = 1;
  }

  malloc(size: number, initialData?: Uint8Array): Slot {
    const buffer = new Uint8Array(size);
    if (initialData) {
      if (initialData.byteLength !== size) {
        throw new Error("initialData size does not match buffer size");
      }
      buffer.set(initialData);
    }

    const slot = this.#nextSlot++;
    this.#buffers.set(slot, { buffer, ref: 1 });
    return slot;
  }

  incRef(slot: Slot): void {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref++;
  }

  decRef(slot: Slot): void {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref--;
    if (buffer.ref === 0) {
      this.#buffers.delete(slot);
    }
  }

  async read(
    slot: Slot,
    start?: number,
    count?: number,
  ): Promise<Uint8Array<ArrayBuffer>> {
    return this.readSync(slot, start, count);
  }

  readSync(
    slot: Slot,
    start?: number,
    count?: number,
  ): Uint8Array<ArrayBuffer> {
    const buffer = this.#getBuffer(slot);
    if (start === undefined) start = 0;
    if (count === undefined) count = buffer.byteLength - start;
    return buffer.slice(start, start + count);
  }

  async prepareKernel(kernel: Kernel): Promise<Executable<void>> {
    return this.prepareKernelSync(kernel);
  }

  prepareKernelSync(kernel: Kernel): Executable<void> {
    return new Executable(kernel, undefined);
  }

  async prepareRoutine(routine: Routine): Promise<Executable> {
    return this.prepareRoutineSync(routine);
  }

  prepareRoutineSync(routine: Routine): Executable {
    return new Executable(routine, undefined);
  }

  dispatch(exe: Executable<void>, inputs: Slot[], outputs: Slot[]): void {
    if (exe.source instanceof Routine) {
      return runCpuRoutine(
        exe.source,
        inputs.map((slot) => this.#getBuffer(slot)),
        outputs.map((slot) => this.#getBuffer(slot)),
      );
    }

    const kernel = exe.source as Kernel;
    let { exp, epilogue } = tuneNullopt(kernel);
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
    const outputBuffers = outputs.map((slot) => this.#getBuffer(slot));

    exp = rewriteSignbitLoads(exp, inputBuffers.length);
    epilogue = epilogue && rewriteSignbitLoads(epilogue, inputBuffers.length);

    const usedArgs = new Map(
      [
        ...exp.collect((exp) => exp.op === AluOp.GlobalIndex),
        ...(epilogue
          ? epilogue.collect((exp) => exp.op === AluOp.GlobalIndex)
          : []),
      ].map((exp) => [exp.arg[0] as number, exp.dtype]),
    );

    const inputArrays: ArrayLike<number>[] = [
      ...inputBuffers.map((buf, i) => {
        const dtype = usedArgs.get(i);
        if (!dtype) return null!; // This arg is unused, so we just blank it out.
        return dtypedArray(dtype, buf);
      }),
      // Byte views of the inputs, from rewritten signbit loads.
      ...inputBuffers,
    ];
    const outputArray = dtypedArray(kernel.dtype, outputBuffers[0]);

    const globals = (gid: number, bufidx: number) => {
      if (gid < 0 || gid >= inputArrays.length)
        throw new Error("gid out of bounds: " + gid);
      if (bufidx < 0 || bufidx >= inputArrays[gid].length)
        throw new Error("bufidx out of bounds: " + bufidx);
      return inputArrays[gid][bufidx];
    };
    if (!kernel.reduction) {
      for (let i = 0; i < kernel.size; i++) {
        outputArray[i] = exp.evaluate({ gidx: i }, globals);
      }
    } else {
      for (let i = 0; i < kernel.size; i++) {
        let acc = kernel.reduction.identity;
        for (let j = 0; j < kernel.reduction.size; j++) {
          const item = exp.evaluate({ gidx: i, ridx: j }, globals);
          acc = kernel.reduction.evaluate(acc, item);
        }
        outputArray[i] = epilogue!.evaluate({ acc, gidx: i }, globals);
      }
    }
  }

  #getBuffer(slot: Slot): Uint8Array<ArrayBuffer> {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return buffer.buffer;
  }
}
