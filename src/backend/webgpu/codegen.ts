// Shared codegen helpers for generating WGSL.

import { erfSrc, threefrySrc } from "./builtins";
import { AluExp, AluGroup, AluOp, DType, isFloatDtype } from "../../alu";
import { UnsupportedOpError } from "../../backend";
import { mapSetUnion, strip1 } from "../../utils";

export interface ShaderInfo {
  code: string; // WGSL shader source code.
  numInputs: number;
  numOutputs: number;
  hasUniform: boolean;
  passes: {
    grid: [number, number]; // Grid size (number of workgroups) in x and y.
    uniform?: Uint8Array<ArrayBuffer>; // Optional uniform value.
  }[];
}

export const headerWgsl = String.raw`
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn inf() -> f32 { let bits = 0x7f800000u; return bitcast<f32>(bits); }
`.trim();

/** Builder class for simple program generation with WGSL. */
export class WgslBuilder {
  readonly pushIndent = Symbol("pushIndent");
  readonly popIndent = Symbol("popIndent");
  readonly lines: string[] = [];
  #indent = "";

  emit(...lines: (string | symbol)[]): void {
    for (const line of lines) {
      if (line === this.pushIndent) this.#indent += "  ";
      else if (line === this.popIndent)
        this.#indent = this.#indent.slice(0, -2);
      else this.lines.push(line ? this.#indent + (line as string) : "");
    }
  }

  emitPreamble(device: GPUDevice, exps: (AluExp | null | undefined)[]): void {
    let hasFloat16 = false;
    let distinctOps = new Map<AluOp, Set<DType>>();
    for (const exp of exps) {
      if (exp == null) continue;
      hasFloat16 ||= exp.some((e) => e.dtype === DType.Float16);
      distinctOps = mapSetUnion(distinctOps, exp.distinctOps());
    }
    if (hasFloat16) {
      if (!device.features.has("shader-f16")) {
        throw new Error("WebGPU device does not support shader-f16 feature");
      }
      this.emit("enable f16;");
    }
    this.emit(headerWgsl);
    if (distinctOps.has(AluOp.Threefry2x32)) {
      this.emit(threefrySrc);
    }
    if (distinctOps.has(AluOp.Erf) || distinctOps.has(AluOp.Erfc)) {
      this.emit(erfSrc);
    }
    this.emit("");
  }

  /**
   * Insert phony assignments, in case some inputs are not in use.
   * <https://github.com/gpuweb/gpuweb/discussions/4582#discussioncomment-9146686>
   */
  emitPhonyAssignments(args: string[]): void {
    if (args.length > 0) {
      this.emit(args.map((arg) => `_ = &${arg};`).join(" "));
    }
  }

  toString(): string {
    return this.lines.join("\n");
  }
}

export function dtypeToWgsl(dtype: DType, storage: boolean = false): string {
  switch (dtype) {
    case DType.Bool:
      return storage ? "i32" : "bool"; // WebGPU does not support bools in buffers.
    case DType.Int8:
      return storage ? "u32" : "i32"; // Packed four-per-u32 in storage.
    case DType.Uint8:
      return "u32"; // Packed four-per-u32 in storage.
    case DType.Int16:
      return storage ? "u32" : "i32"; // Packed two-per-u32 in storage.
    case DType.Uint16:
      return "u32"; // Packed two-per-u32 in storage.
    case DType.Int32:
      return "i32";
    case DType.Uint32:
      return "u32"; // WebGPU supports uint32 in buffers.
    case DType.Float32:
      return "f32";
    case DType.Float16:
      return "f16";
    default:
      throw new Error(`Unsupported dtype for WebGPU: ${dtype}`);
  }
}

export function storageDtypeToWgsl(
  dtype: DType,
  access: "read" | "read_write",
): string {
  if (
    access === "read_write" &&
    (dtype === DType.Int8 ||
      dtype === DType.Uint8 ||
      dtype === DType.Int16 ||
      dtype === DType.Uint16)
  ) {
    return "atomic<u32>";
  }
  return dtypeToWgsl(dtype, true);
}

function packedIntBits(dtype: DType): 8 | 16 | null {
  if (dtype === DType.Int8 || dtype === DType.Uint8) return 8;
  if (dtype === DType.Int16 || dtype === DType.Uint16) return 16;
  return null;
}

function isPackedIntDtype(dtype: DType): boolean {
  return packedIntBits(dtype) !== null;
}

function isSignedPackedIntDtype(dtype: DType): boolean {
  return dtype === DType.Int8 || dtype === DType.Int16;
}

function normalizeWgsl(dtype: DType, value: string): string {
  const bits = packedIntBits(dtype);
  if (bits === null) return value;
  const mask = bits === 8 ? "0xffu" : "0xffffu";
  if (isSignedPackedIntDtype(dtype)) {
    const shift = 32 - bits;
    return `(bitcast<i32>((bitcast<u32>(i32(${value})) & ${mask}) << ${shift}u) >> ${shift})`;
  }
  if (dtype === DType.Uint8 || dtype === DType.Uint16)
    return `(${value} & ${mask})`;
  return value;
}

export function readStorageWgsl(
  dtype: DType,
  buffer: string,
  index: string,
): string {
  const bitWidth = packedIntBits(dtype);
  if (bitWidth !== null) {
    const wordShift = bitWidth === 8 ? "2u" : "1u";
    const laneMask = bitWidth === 8 ? "3u" : "1u";
    const valueMask = bitWidth === 8 ? "0xffu" : "0xffffu";
    const idx = `u32(${index})`;
    const bits = `((${buffer}[${idx} >> ${wordShift}] >> ((${idx} & ${laneMask}) * ${bitWidth}u)) & ${valueMask})`;
    if (isSignedPackedIntDtype(dtype)) {
      const shift = 32 - bitWidth;
      return `(bitcast<i32>(${bits} << ${shift}u) >> ${shift})`;
    }
    return bits;
  }
  const source = `${buffer}[${index}]`;
  if (dtype === DType.Bool) return `(${source} != 0)`;
  return source;
}

export function emitStorageStoreWgsl(
  wb: WgslBuilder,
  dtype: DType,
  buffer: string,
  index: string,
  value: string,
  valueDtype: DType = dtype,
): void {
  const bitWidth = packedIntBits(dtype);
  if (bitWidth !== null) {
    const wordShift = bitWidth === 8 ? "2u" : "1u";
    const laneMask = bitWidth === 8 ? "3u" : "1u";
    const valueMask = bitWidth === 8 ? "0xffu" : "0xffffu";
    const valueBits =
      isSignedPackedIntDtype(dtype) ? `bitcast<u32>(i32(${value}))` : value;
    wb.emit(
      "{",
      wb.pushIndent,
      `let packed_index: u32 = u32(${index});`,
      `let word_index: u32 = packed_index >> ${wordShift};`,
      `let shift: u32 = (packed_index & ${laneMask}) * ${bitWidth}u;`,
      `let mask: u32 = ${valueMask} << shift;`,
      `let bits: u32 = (${valueBits} & ${valueMask}) << shift;`,
      `var old_value: u32 = atomicLoad(&${buffer}[word_index]);`,
      "loop {",
      wb.pushIndent,
      "let next_value: u32 = (old_value & ~mask) | bits;",
      `let exchange = atomicCompareExchangeWeak(&${buffer}[word_index], old_value, next_value);`,
      "if (exchange.exchanged) { break; }",
      "old_value = exchange.old_value;",
      wb.popIndent,
      "}",
      wb.popIndent,
      "}",
    );
    return;
  }

  let rhs = value;
  const storageTy = dtypeToWgsl(dtype, true);
  if (storageTy !== dtypeToWgsl(valueDtype)) rhs = `${storageTy}(${rhs})`;
  wb.emit(`${buffer}[${index}] = ${rhs};`);
}

export function minValueWgsl(dtype: DType): string {
  switch (dtype) {
    case DType.Bool:
      return "0"; // Using i32 representation.
    case DType.Int8:
      return "-128";
    case DType.Uint8:
      return "0u";
    case DType.Int16:
      return "-32768";
    case DType.Uint16:
      return "0u";
    case DType.Int32:
      return "-2147483648"; // -2^31
    case DType.Uint32:
      return "0u";
    case DType.Float32:
      return "-inf()";
    case DType.Float16:
      return "f16(-inf())";
    default:
      throw new Error(`Unsupported dtype for WebGPU: ${dtype}`);
  }
}

export function maxValueWgsl(dtype: DType): string {
  switch (dtype) {
    case DType.Bool:
      return "1"; // Using i32 representation.
    case DType.Int8:
      return "127";
    case DType.Uint8:
      return "255u";
    case DType.Int16:
      return "32767";
    case DType.Uint16:
      return "65535u";
    case DType.Int32:
      return "2147483647"; // 2^31 - 1
    case DType.Uint32:
      return "4294967295u"; // 2^32 - 1
    case DType.Float32:
      return "inf()";
    case DType.Float16:
      return "f16(inf())";
    default:
      throw new Error(`Unsupported dtype for WebGPU: ${dtype}`);
  }
}

export function constToWgsl(dtype: DType, value: any): string {
  if (dtype === DType.Bool) return value ? "true" : "false";
  if (dtype === DType.Int8) return value.toString();
  if (dtype === DType.Uint8) return value.toString() + "u";
  if (dtype === DType.Int16) return value.toString();
  if (dtype === DType.Uint16) return value.toString() + "u";
  if (dtype === DType.Int32) return value.toString();
  if (dtype === DType.Uint32) return value.toString() + "u"; // WebGPU uses 'u' suffix for uint32.
  if (dtype === DType.Float32) {
    if (Number.isNaN(value)) return "nan()";
    if (!Number.isFinite(value)) return value > 0 ? "inf()" : "-inf()";
    return "f32(" + value.toString() + ")";
  }
  if (dtype === DType.Float16) {
    if (Number.isNaN(value)) return "f16(nan())";
    if (!Number.isFinite(value))
      return value > 0 ? "f16(inf())" : "f16(-inf())";
    return "f16(" + value.toString() + ")";
  }
  throw new Error(`Unsupported const dtype: ${dtype}`);
}

export function reduceOpWgsl(
  op: AluOp,
  dtype: DType,
  a: string,
  b: string,
): string {
  if (op === AluOp.Add) return `(${a} + ${b})`;
  if (op === AluOp.Mul) return `(${a} * ${b})`;
  if (op === AluOp.Min)
    return dtype === DType.Bool ? `(${a} && ${b})` : `min(${a}, ${b})`;
  if (op === AluOp.Max)
    return dtype === DType.Bool ? `(${a} || ${b})` : `max(${a}, ${b})`;
  throw new Error(`Unsupported reduction op: ${op}`);
}

/** Codegen for WebGPU expressions, linearizing AluOp into a kernel. */
export class WgslExpCodegen {
  #gensymCount = 0;
  #references = new Map<AluExp, number>();
  #seen = new Set<AluExp>();
  #context = new Map<AluExp, string>();

  constructor(
    readonly wb: WgslBuilder,
    readonly args: string[],
  ) {}

  #gensym() {
    return `alu${this.#gensymCount++}`;
  }

  #isGensym(text: string) {
    return text.match(/^alu[0-9]+$/);
  }

  /**
   * Count references for an expression.
   *
   * Used to get an ahead-of-time reference count for each node in the AluExp.
   * Expressions with reference count greater than 1 are stored in temporary
   * variables to avoid recomputation.
   */
  countReferences(exp: AluExp): void {
    this.#references.set(exp, (this.#references.get(exp) ?? 0) + 1);
    if (!this.#seen.has(exp)) {
      this.#seen.add(exp);
      for (const src of exp.src) this.countReferences(src);
    }
  }

  reset(): void {
    this.#references.clear();
    this.#seen.clear();
    this.#context.clear();
  }

  /**
   * Generate code for an expression.
   *
   * Calls itself recursively and eliminates common subexpressions by storing
   * them in temporary variables, emitted to the current builder scope. This is
   * a side-effect that leads to multiline code generation.
   */
  run(exp: AluExp): string {
    if (this.#context.has(exp)) return this.#context.get(exp)!;
    const { op, src, dtype, arg } = exp;

    // Some of these cases early `return` to force-inline them.
    let source = "";
    if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
      const a = this.run(src[0]);
      const b = this.run(src[1]);
      if (op === AluOp.Add) {
        if (dtype === DType.Bool) source = `(${a} || ${b})`;
        else source = `(${a} + ${b})`;
      } else if (op === AluOp.Sub) source = `(${a} - ${b})`;
      else if (op === AluOp.Mul) {
        if (dtype === DType.Bool) source = `(${a} && ${b})`;
        else source = `(${a} * ${b})`;
      } else if (op === AluOp.Idiv)
        source = isFloatDtype(dtype) ? `trunc(${a} / ${b})` : `(${a} / ${b})`;
      else if (op === AluOp.Mod) source = `(${a} % ${b})`;
      else if (op === AluOp.Min) {
        if (dtype === DType.Bool) source = `(${a} && ${b})`;
        else source = `min(${strip1(a)}, ${strip1(b)})`;
      } else if (op === AluOp.Max) {
        if (dtype === DType.Bool) source = `(${a} || ${b})`;
        else source = `max(${strip1(a)}, ${strip1(b)})`;
      } else if (op === AluOp.BitCombine) {
        if (arg === "and") source = `(${a} & ${b})`;
        else if (arg === "or") source = `(${a} | ${b})`;
        else source = dtype === DType.Bool ? `(${a} != ${b})` : `(${a} ^ ${b})`;
      } else if (op === AluOp.BitShift) {
        if (arg === "shl") source = `(${a} << ${b})`;
        else source = `(${a} >> ${b})`;
      } else if (op === AluOp.Cmplt) source = `(${a} < ${b})`;
      else if (op === AluOp.Cmpne) {
        // Edge case: WebGPU doesn't handle NaN correctly, it's unspecified.
        // This is a reliable way I found to detect NaNs, since the spec says
        // for `max()`: if one operand is a NaN, the other is returned.
        if (isFloatDtype(src[0].dtype)) {
          const x = this.#isGensym(a) ? a : this.#gensym();
          if (x !== a) this.wb.emit(`let ${x} = ${a};`);
          source = `(${x} != ${b} || min(${x}, ${dtypeToWgsl(src[0].dtype)}(inf())) != ${x})`;
        } else {
          source = `(${a} != ${b})`;
        }
      }
      if (
        isPackedIntDtype(dtype) &&
        (op === AluOp.Add ||
          op === AluOp.Sub ||
          op === AluOp.Mul ||
          op === AluOp.Idiv ||
          op === AluOp.Mod ||
          op === AluOp.BitCombine ||
          op === AluOp.BitShift)
      ) {
        source = normalizeWgsl(dtype, source);
      }
    } else if (AluGroup.Unary.has(op)) {
      if (op === AluOp.Reciprocal && src[0].op === AluOp.Sqrt) {
        // Special case: 1/sqrt(x) is optimized as rsqrt(x)
        const a = this.run(src[0].src[0]);
        source = `inverseSqrt(${a})`;
      } else {
        const a = this.run(src[0]);
        if (op === AluOp.Sin) source = `sin(${strip1(a)})`;
        else if (op === AluOp.Cos) source = `cos(${strip1(a)})`;
        else if (op === AluOp.Asin) source = `asin(${strip1(a)})`;
        else if (op === AluOp.Atan) source = `atan(${strip1(a)})`;
        else if (op === AluOp.Exp) source = `exp(${strip1(a)})`;
        else if (op === AluOp.Log) source = `log(${strip1(a)})`;
        else if (op === AluOp.Erf || op === AluOp.Erfc) {
          const funcName = op === AluOp.Erf ? "erf" : "erfc";
          if (dtype !== DType.Float32) {
            // Always compute special functions in f32 for precision.
            source = `${dtypeToWgsl(dtype)}(${funcName}(f32(${strip1(a)})))`;
          } else {
            source = `${funcName}(${strip1(a)})`;
          }
        } else if (op === AluOp.Sqrt) source = `sqrt(${strip1(a)})`;
        else if (op === AluOp.Reciprocal) source = `(1.0 / ${a})`;
        else if (op === AluOp.Floor) source = `floor(${strip1(a)})`;
        else if (op === AluOp.Ceil) source = `ceil(${strip1(a)})`;
        else if (op === AluOp.Cast) {
          const srcTy = dtypeToWgsl(src[0].dtype);
          const dstTy = dtypeToWgsl(dtype);
          if (
            isPackedIntDtype(dtype) &&
            isFloatDtype(src[0].dtype)
          ) {
            const minVal = minValueWgsl(dtype);
            const maxVal = maxValueWgsl(dtype);
            const x = this.#isGensym(a) ? a : this.#gensym();
            if (x !== a) this.wb.emit(`let ${x}: ${srcTy} = ${strip1(a)};`);
            source = `${dstTy}(clamp(${x}, ${srcTy}(${minVal}), ${srcTy}(${maxVal})))`;
          } else if (
            isFloatDtype(src[0].dtype) &&
            !(isFloatDtype(dtype) || dtype === DType.Bool)
          ) {
            // Edge case: Float->Int conversion with saturating upper bound.
            const maxVal = maxValueWgsl(dtype);
            const x = this.#isGensym(a) ? a : this.#gensym();
            if (x !== a) this.wb.emit(`let ${x}: ${srcTy} = ${strip1(a)};`);
            source = `select(${dstTy}(${x}), ${maxVal}, ${x} >= ${srcTy}(${maxVal}))`;
          } else {
            source = `${dstTy}(${strip1(a)})`;
          }
          source = normalizeWgsl(dtype, source);
        } else if (op === AluOp.Bitcast)
          source = `bitcast<${dtypeToWgsl(dtype)}>(${strip1(a)})`;
        if (op === AluOp.Bitcast) source = normalizeWgsl(dtype, source);
      }
    } else if (op === AluOp.Where) {
      // select(f, t, cond) -> cond ? t : f
      source = `select(${strip1(this.run(src[2]))}, ${strip1(this.run(src[1]))}, ${strip1(this.run(src[0]))})`;
    } else if (op === AluOp.Threefry2x32) {
      const x = this.#gensym(); // temporary to hold the `vec2<u32>(x0, x1)`
      const [k0, k1, c0, c1] = src.map((x) => strip1(this.run(x)));
      this.wb.emit(
        `let ${x} = threefry2x32(vec2(${k0}, ${k1}), vec2(${c0}, ${c1}));`,
      );
      if (arg === "xor") source = `(${x}.x ^ ${x}.y)`;
      else if (arg === 0) source = `${x}.x`;
      else if (arg === 1) source = `${x}.y`;
      else throw new UnsupportedOpError(op, dtype, "webgpu", arg);
    } else if (op === AluOp.Const) {
      return constToWgsl(dtype, arg);
    } else if (op === AluOp.Special) {
      return arg[0] as string;
    } else if (op === AluOp.Variable) {
      return arg as string;
    } else if (op === AluOp.GlobalIndex) {
      source = readStorageWgsl(
        dtype,
        this.args[arg[0]],
        strip1(this.run(src[0])),
      );
    }

    if (!source) throw new UnsupportedOpError(op, dtype, "webgpu", arg);
    const typeName = dtypeToWgsl(dtype);
    if ((this.#references.get(exp) ?? 0) > 1) {
      const name = this.#gensym();
      this.#context.set(exp, name);
      this.wb.emit(`let ${name}: ${typeName} = ${strip1(source)};`);
      return name;
    } else {
      this.#context.set(exp, source);
      return source;
    }
  }
}

export const gridOffsetY = 16384;

export function calculateGrid(gridSize: number): [number, number] {
  let gridX = gridSize;
  let gridY = 1;
  // https://web3dsurvey.com/webgpu/limits/maxComputeWorkgroupsPerDimension
  // device.limits.maxComputeWorkgroupsPerDimension = 65535
  if (gridSize > 65535) {
    gridX = gridOffsetY;
    gridY = Math.ceil(gridSize / gridOffsetY);
  }
  return [gridX, gridY];
}
