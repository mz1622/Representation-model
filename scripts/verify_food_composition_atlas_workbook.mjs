#!/usr/bin/env node
/** Render a compact visual verification for the shared Atlas workbook template. */

import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath = process.argv[2];
const previewPath = process.argv[3];

if (!workbookPath || !previewPath) {
  throw new Error("Usage: verify_food_composition_atlas_workbook.mjs <workbook.xlsx> <preview.png>");
}

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const inspection = await workbook.inspect({
  kind: "region",
  sheetId: "Instructions",
  range: "A1:H18",
  maxChars: 1600,
});
if (!inspection) throw new Error("Could not inspect the Instructions sheet.");
const preview = await workbook.render({ sheetName: "Instructions", autoCrop: "all", scale: 1.2, format: "png" });
await fs.mkdir(path.dirname(previewPath), { recursive: true });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
console.log(`Rendered ${previewPath}`);
