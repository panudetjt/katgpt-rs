/* tslint:disable */
/* eslint-disable */

/**
 * Result of identifying a file with Magika.
 */
export class MagikaResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Export result as a JSON string.
     */
    to_json(): string;
    /**
     * Human-readable description (e.g. "Portable Document Format").
     */
    readonly description: string;
    /**
     * Common file extensions as a comma-separated string (e.g. "pdf" or "jpg,jpeg").
     */
    readonly extensions: string;
    /**
     * High-level content group (e.g. "document", "image", "code", "archive").
     */
    readonly group: string;
    /**
     * True if content is text-based.
     */
    readonly is_text: boolean;
    /**
     * Canonical content type label (e.g. "pdf", "python", "png", "txt", "unknown").
     */
    readonly label: string;
    /**
     * Official MIME type (e.g. "application/pdf", "image/png").
     */
    readonly mime_type: string;
    /**
     * Model prediction score in `[0.0, 1.0]`.
     */
    readonly score: number;
}

/**
 * Identifies a file's content type from a Uint8Array byte slice.
 */
export function identify(bytes: Uint8Array): MagikaResult;

/**
 * Identifies a file and directly returns a JSON string representation.
 */
export function identify_json(bytes: Uint8Array): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_magikaresult_free: (a: number, b: number) => void;
    readonly identify: (a: number, b: number) => number;
    readonly identify_json: (a: number, b: number) => [number, number];
    readonly magikaresult_description: (a: number) => [number, number];
    readonly magikaresult_extensions: (a: number) => [number, number];
    readonly magikaresult_group: (a: number) => [number, number];
    readonly magikaresult_is_text: (a: number) => number;
    readonly magikaresult_label: (a: number) => [number, number];
    readonly magikaresult_mime_type: (a: number) => [number, number];
    readonly magikaresult_score: (a: number) => number;
    readonly magikaresult_to_json: (a: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
