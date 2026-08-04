import { expect, suite, test } from "vitest";

import { strip1 } from "./utils";

suite("strip1()", () => {
  test("strips a matched outer parenthesis pair", () => {
    expect(strip1("(a + b)")).toBe("a + b");
    expect(strip1("((a) + (b))")).toBe("(a) + (b)");
    expect(strip1("(x != b || min(x, f32(inf())) != x)")).toBe(
      "x != b || min(x, f32(inf())) != x",
    );
  });

  test("leaves expressions without outer parentheses unchanged", () => {
    expect(strip1("a + b")).toBe("a + b");
    expect(strip1("alu0")).toBe("alu0");
    expect(strip1("min(a, b)")).toBe("min(a, b)");
    expect(strip1("f32(1)")).toBe("f32(1)");
    expect(strip1("f32(x) + g(y)")).toBe("f32(x) + g(y)");
  });

  test("does not strip when the outer parentheses are not a matched pair", () => {
    expect(strip1("(a) + (b)")).toBe("(a) + (b)");
    expect(strip1("(a + b) * (c + d)")).toBe("(a + b) * (c + d)");
    expect(strip1("(a || b) && (c || d)")).toBe("(a || b) && (c || d)");
  });
});
