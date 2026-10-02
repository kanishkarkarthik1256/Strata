/**
 * Theme contract.
 *
 * The theme is applied as `data-theme` on <html>, which is what the
 * stylesheet and the viewer's backdrop key off. These tests pin the three
 * properties the rest of the app depends on: a persisted choice wins,
 * "system" follows the OS setting, and every change is actually applied to
 * the document rather than only held in React state.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { ThemeProvider, THEME_STORAGE_KEY, useTheme } from "./useTheme";
import type { ReactNode } from "react";

function setSystemPrefersLight(prefersLight: boolean) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: query.includes("light") ? prefersLight : !prefersLight,
    media: query,
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })) as unknown as typeof window.matchMedia;
}

const wrapper = ({ children }: { children: ReactNode }) => (
  <ThemeProvider>{children}</ThemeProvider>
);

describe("useTheme", () => {
  beforeEach(() => {
    localStorage.clear();
    setSystemPrefersLight(false);
  });

  afterEach(() => {
    delete document.documentElement.dataset.theme;
  });

  it("follows the OS setting when nothing has been chosen", () => {
    setSystemPrefersLight(true);
    const { result } = renderHook(() => useTheme(), { wrapper });

    expect(result.current.choice).toBe("system");
    expect(result.current.resolved).toBe("light");
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("applies and persists an explicit choice", () => {
    const { result } = renderHook(() => useTheme(), { wrapper });
    expect(document.documentElement.dataset.theme).toBe("dark");

    act(() => result.current.setChoice("light"));

    expect(result.current.resolved).toBe("light");
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe("light");
  });

  it("toggles between light and dark and leaves the system setting behind", () => {
    const { result } = renderHook(() => useTheme(), { wrapper });
    expect(result.current.choice).toBe("system");

    act(() => result.current.toggle());

    // System resolved to dark, so the first toggle pins light.
    expect(result.current.choice).toBe("light");
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe("light");

    act(() => result.current.toggle());
    expect(result.current.resolved).toBe("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("restores a stored choice over the OS setting", () => {
    localStorage.setItem(THEME_STORAGE_KEY, "light");
    setSystemPrefersLight(false);

    const { result } = renderHook(() => useTheme(), { wrapper });

    expect(result.current.choice).toBe("light");
    expect(result.current.resolved).toBe("light");
  });
});
