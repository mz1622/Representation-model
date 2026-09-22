#!/usr/bin/env node
/** Build the five read-only Food Composition Atlas review workbooks. */

import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const DECISIONS = [
  "accept",
  "reject",
  "keep separate",
  "needs evidence",
  "defer",
];

const CONFLICT_DECISIONS = [
  "retain all source values",
  "eligible for derived-release selection",
  "invalidate source record",
];

const queueDefinitions = [
  {
    key: "axis_mapping_review",
    filename: "Axis_Mapping_Review.xlsx",
    title: "Axis Mapping Review",
    purpose: "Review candidate mappings between source-native composition labels and proposed composition axes. A decision never authorizes numerical pooling by itself.",
  },
  {
    key: "food_identity_review",
    filename: "Food_Identity_Review.xlsx",
    title: "Food Identity Review",
    purpose: "Review exact-name candidate groups and identity facets. Exact naming is a candidate signal, not a cross-source food merge decision.",
  },
  {
    key: "cross_source_conflict_review",
    filename: "Cross_Source_Conflict_Review.xlsx",
    title: "Cross-Source Conflict Review",
    purpose: "Review retained source-specific disagreements. The default decision is to retain all source values; selecting a source winner requires separate documented governance.",
    isConflict: true,
  },
  {
    key: "measurement_exception_review",
    filename: "Measurement_Exception_Review.xlsx",
    title: "Measurement Exception Review",
    purpose: "Review censored, range-only, trace, invalid, or non-exact conversion expressions. Missing values must never be changed into zeros.",
    // The full exception queue is retained row-for-row. The workbook exposes
    // decision-relevant evidence; its immutable CSV contains every source
    // field and is linked from the Atlas for record-level follow-up.
    workbookColumns: [
      "measurement_id",
      "source_key",
      "food_observation_id",
      "component_observation_id",
      "target_axis_id",
      "raw_value",
      "raw_unit",
      "raw_basis",
      "value_status",
      "conversion_status",
      "measurement_modality",
      "source_measurement_locator",
      "exact_name_group_id",
      "review_type",
      "recommended_decision",
    ],
  },
  {
    key: "source_registry_review",
    filename: "Source_Registry_Review.xlsx",
    title: "Source Registry Review",
    purpose: "Review source metadata, access conditions, versioning, and source dependencies. This does not change the raw source archive.",
  },
];

const colors = {
  navy: "#17324D",
  blue: "#216A94",
  pale: "#EAF2F8",
  light: "#F7F9FB",
  line: "#CBD5E1",
  amber: "#FFF2CC",
  text: "#18212F",
};

function parseArgs(argv) {
  const options = { atlas: null, output: null, queue: null };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--atlas") options.atlas = argv[++index];
    else if (value === "--output") options.output = argv[++index];
    else if (value === "--queue") options.queue = argv[++index];
    else throw new Error(`Unknown argument: ${value}`);
  }
  if (!options.atlas || !options.output) {
    throw new Error("Usage: build_food_composition_atlas_workbooks.mjs --atlas <atlas-output-dir> --output <workbook-dir> [--queue <queue-key>]");
  }
  return options;
}

function parseCsvHeader(text) {
  const firstLine = text.slice(0, text.indexOf("\n") >= 0 ? text.indexOf("\n") : text.length).replace(/\r$/, "");
  const cells = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < firstLine.length; index += 1) {
    const character = firstLine[index];
    if (character === '"') {
      if (quoted && firstLine[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      cells.push(value);
      value = "";
    } else {
      value += character;
    }
  }
  cells.push(value);
  return cells;
}

function parseCsvRows(text) {
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      row.push(value);
      value = "";
    } else if (character === "\n" && !quoted) {
      row.push(value.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      value = "";
    } else {
      value += character;
    }
  }
  if (value.length || row.length) {
    row.push(value.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows;
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function selectWorkbookColumns(csvText, requestedColumns) {
  if (!requestedColumns) return csvText;
  const rows = parseCsvRows(csvText);
  const header = rows.shift();
  if (!header) throw new Error("Review queue CSV has no header row.");
  const indexes = requestedColumns.map((column) => {
    const index = header.indexOf(column);
    if (index < 0) throw new Error(`Review queue lacks requested workbook field: ${column}`);
    return index;
  });
  const selected = [requestedColumns, ...rows.map((row) => indexes.map((index) => row[index] ?? ""))];
  return selected.map((row) => row.map(csvCell).join(",")).join("\n") + "\n";
}

function columnName(index) {
  let number = index + 1;
  let result = "";
  while (number > 0) {
    const remainder = (number - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    number = Math.floor((number - 1) / 26);
  }
  return result;
}

function styleTitle(sheet, title, subtitle) {
  sheet.showGridLines = false;
  sheet.getRange("A1:H1").merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1:H1").format = {
    fill: colors.navy,
    font: { color: "#FFFFFF", bold: true, size: 16, name: "Arial" },
    horizontalAlignment: "left",
    verticalAlignment: "center",
  };
  sheet.getRange("A1:H1").format.rowHeight = 28;
  sheet.getRange("A3:H3").merge();
  sheet.getRange("A3").values = [[subtitle]];
  sheet.getRange("A3:H3").format = {
    font: { color: colors.text, italic: true, size: 10, name: "Arial" },
    wrapText: true,
    verticalAlignment: "top",
  };
  sheet.getRange("A3:H3").format.rowHeight = 42;
  sheet.getRange("A:A").format.columnWidth = 29;
  sheet.getRange("B:B").format.columnWidth = 58;
  sheet.getRange("C:H").format.columnWidth = 18;
}

function styleHeader(range) {
  range.format = {
    fill: colors.blue,
    font: { color: "#FFFFFF", bold: true, size: 10, name: "Arial" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: colors.line },
  };
  range.format.rowHeight = 32;
}

async function addInstructions(workbook, definition, manifest, sourceRows) {
  const sheet = workbook.worksheets.add("Instructions");
  styleTitle(sheet, definition.title, definition.purpose);
  const rows = [
    ["Dataset snapshot", manifest.atlas_version],
    ["Build timestamp (UTC)", manifest.built_at_utc.replace("T", " ").replace("+00:00", " UTC")],
    ["Queue", definition.key],
    ["Queue row count", manifest.review_queues[definition.key].rows],
    ["Queue SHA-256", manifest.review_queues[definition.key].sha256],
    ["", ""],
    ["How to review", "Use one decision for each reviewed record. Record the reviewer, date, evidence URL, and rationale. Do not overwrite raw source fields."],
    ["Missing values", "Not reported or unknown. Missing is never a numeric zero."],
    ["Source values", "Source-specific values remain separate. A mapping decision does not authorize numerical pooling."],
    ["Identity", "Exact-name groups and search candidates are not confirmed cross-source food merges."],
    ["Submission check", "Use the exact dataset snapshot above. Do not submit decisions with unknown IDs, unsupported vocabulary, or a source-winner decision without a rationale."],
  ];
  sheet.getRangeByIndexes(4, 0, rows.length, 2).values = rows;
  sheet.getRange("A5:A15").format = { fill: colors.pale, font: { bold: true, name: "Arial" }, verticalAlignment: "top" };
  sheet.getRange("B5:B15").format = { font: { name: "Arial", size: 10 }, wrapText: true, verticalAlignment: "top" };
  sheet.getRange("A5:B15").format.borders = { preset: "outside", style: "thin", color: colors.line };
  sheet.getRange("A5:B15").format.autofitRows();
  sheet.getRange("A17:H17").merge();
  sheet.getRange("A17").values = [["Source reference"]];
  sheet.getRange("A17:H17").format = { fill: colors.pale, font: { bold: true, name: "Arial" } };
  const compactSources = sourceRows.slice(0, 8).map((source) => [source.display_name || source.source_key, source.official_url || "Not recorded"]);
  if (compactSources.length) {
    sheet.getRangeByIndexes(17, 0, compactSources.length, 2).values = compactSources;
    sheet.getRangeByIndexes(17, 0, compactSources.length, 2).format = { font: { name: "Arial", size: 10 }, wrapText: true };
  }
  sheet.freezePanes.freezeRows(4);
  return sheet;
}

async function addSnapshot(workbook, manifest, definition) {
  const sheet = workbook.worksheets.add("Dataset Snapshot");
  styleTitle(sheet, "Dataset Snapshot", "Immutable build metadata for this review workbook. The snapshot identifies the Atlas inputs and does not select or pool source values.");
  const counts = manifest.entity_counts;
  const metricRows = [
    ["Registered sources", counts.registered_sources],
    ["Numerically integrated sources", counts.numerically_integrated_sources],
    ["Source food observations", counts.source_food_observations],
    ["Exact-name candidate groups", counts.exact_name_candidate_groups],
    ["Mapped numeric measurements", counts.mapped_numeric_measurements],
    ["Direct-mass measurements", counts.direct_mass_measurements],
    ["Retained cross-source conflicts", counts.cross_source_conflicts],
    ["Proposed prediction axes", counts.proposed_prediction_axes],
    ["Review queue", definition.key],
    ["Review queue SHA-256", manifest.review_queues[definition.key].sha256],
  ];
  sheet.getRange("A5:B5").values = [["Metric", "Value"]];
  styleHeader(sheet.getRange("A5:B5"));
  sheet.getRangeByIndexes(5, 0, metricRows.length, 2).values = metricRows;
  sheet.getRangeByIndexes(5, 0, metricRows.length, 2).format = { font: { name: "Arial", size: 10 }, verticalAlignment: "center" };
  sheet.getRange("A5:B15").format.borders = { preset: "outside", style: "thin", color: colors.line };
  const inputs = manifest.input_files.map((item) => [item.label, item.path, item.sha256]);
  sheet.getRange("D5:F5").values = [["Immutable input", "Recorded path", "SHA-256"]];
  styleHeader(sheet.getRange("D5:F5"));
  sheet.getRangeByIndexes(5, 3, inputs.length, 3).values = inputs;
  sheet.getRangeByIndexes(5, 3, inputs.length, 3).format = { font: { name: "Arial", size: 9 }, wrapText: true, verticalAlignment: "top" };
  sheet.getRangeByIndexes(4, 3, inputs.length + 1, 3).format.borders = { preset: "outside", style: "thin", color: colors.line };
  sheet.getRange("A:A").format.columnWidth = 31;
  sheet.getRange("B:B").format.columnWidth = 32;
  sheet.getRange("D:D").format.columnWidth = 29;
  sheet.getRange("E:E").format.columnWidth = 42;
  sheet.getRange("F:F").format.columnWidth = 55;
  sheet.freezePanes.freezeRows(5);
  return sheet;
}

async function addVocabulary(workbook, definition) {
  const sheet = workbook.worksheets.add("Decision Vocabulary");
  styleTitle(sheet, "Decision Vocabulary", "Permitted review decisions. A reviewer must provide a rationale and an evidence URL for any decision other than defer.");
  const rows = [
    ["Decision", "Use"],
    ["accept", "The proposed mapping, identity statement, source metadata, or exception interpretation is supported by cited evidence."],
    ["reject", "The proposed item is unsupported or incorrect. State the reason and preserve the raw record."],
    ["keep separate", "The records or expressions are related but should not be merged or treated as the same identity."],
    ["needs evidence", "A decision cannot be made without additional authoritative evidence."],
    ["defer", "Set aside for a later review cycle without changing the underlying data."],
  ];
  if (definition.isConflict) {
    rows.push(
      ["retain all source values", "Default for a retained cross-source disagreement. Values continue to be displayed in parallel."],
      ["eligible for derived-release selection", "A documented governance process may consider a derived release. This workbook does not select a source winner."],
      ["invalidate source record", "Use only with an explicit, cited reason that a source record should be excluded from a derived release."],
    );
  }
  sheet.getRangeByIndexes(4, 0, rows.length, 2).values = rows;
  styleHeader(sheet.getRange("A5:B5"));
  sheet.getRange("A6:A20").format = { fill: colors.pale, font: { bold: true, name: "Arial" }, verticalAlignment: "top" };
  sheet.getRange("B6:B20").format = { font: { name: "Arial", size: 10 }, wrapText: true, verticalAlignment: "top" };
  sheet.getRange(`A5:B${4 + rows.length}`).format.borders = { preset: "outside", style: "thin", color: colors.line };
  sheet.getRange("A:A").format.columnWidth = 35;
  sheet.getRange("B:B").format.columnWidth = 85;
  return sheet;
}

async function addEvidenceLinks(workbook, sourceRows) {
  const sheet = workbook.worksheets.add("Evidence Links");
  styleTitle(sheet, "Evidence Links", "Source registry links included with this snapshot. Reviewers should cite direct authority or source evidence in the Review Queue rather than relying on a source name alone.");
  const rows = [["Source", "Source key", "Official URL", "License or terms", "Access status", "Version"]];
  for (const source of sourceRows) {
    rows.push([
      source.display_name || "Not recorded",
      source.source_key || "Not recorded",
      source.official_url || "Not recorded",
      source.license_or_terms || "Not recorded",
      source.access_status || "Not recorded",
      source.source_version || "Not recorded",
    ]);
  }
  sheet.getRangeByIndexes(4, 0, rows.length, 6).values = rows;
  styleHeader(sheet.getRange("A5:F5"));
  sheet.getRangeByIndexes(5, 0, rows.length - 1, 6).format = { font: { name: "Arial", size: 9 }, wrapText: true, verticalAlignment: "top" };
  sheet.getRange(`A5:F${4 + rows.length}`).format.borders = { preset: "outside", style: "thin", color: colors.line };
  sheet.getRange("A:A").format.columnWidth = 30;
  sheet.getRange("B:B").format.columnWidth = 19;
  sheet.getRange("C:C").format.columnWidth = 55;
  sheet.getRange("D:D").format.columnWidth = 36;
  sheet.getRange("E:E").format.columnWidth = 23;
  sheet.getRange("F:F").format.columnWidth = 20;
  sheet.freezePanes.freezeRows(5);
  return sheet;
}

async function decorateQueue(workbook, csvText, definition, manifest) {
  const sheet = workbook.worksheets.getItem("Review Queue");
  sheet.showGridLines = false;

  const headers = parseCsvHeader(csvText);
  const originalColumns = headers.length;
  const rows = manifest.review_queues[definition.key].rows;
  const firstDataRow = 2;
  const finalDataRow = firstDataRow + rows - 1;
  const extraHeaders = ["reviewer", "review_date", "decision", "rationale", "evidence_url", "follow_up_needed"];
  sheet.getRangeByIndexes(0, originalColumns, 1, extraHeaders.length).values = [extraHeaders];
  const finalLastColumn = columnName(originalColumns + extraHeaders.length - 1);
  styleHeader(sheet.getRange(`A1:${finalLastColumn}1`));
  const reviewStart = columnName(originalColumns);
  const reviewEnd = columnName(originalColumns + extraHeaders.length - 1);
  sheet.getRange(`${reviewStart}${firstDataRow}:${reviewEnd}${finalDataRow}`).format.fill = colors.amber;
  sheet.getRange(`${reviewStart}1:${reviewEnd}1`).format.fill = colors.blue;
  sheet.getRange(`${reviewStart}${firstDataRow}:${reviewEnd}${finalDataRow}`).format.wrapText = true;
  const decisionColumn = columnName(originalColumns + 2);
  const choices = definition.isConflict ? [...DECISIONS, ...CONFLICT_DECISIONS] : DECISIONS;
  sheet.getRange(`${decisionColumn}${firstDataRow}:${decisionColumn}${finalDataRow}`).dataValidation = {
    rule: { type: "list", values: choices },
  };
  const followUpColumn = columnName(originalColumns + 5);
  sheet.getRange(`${followUpColumn}${firstDataRow}:${followUpColumn}${finalDataRow}`).dataValidation = {
    rule: { type: "list", values: ["yes", "no"] },
  };
  sheet.getRange("A:A").format.columnWidth = 22;
  for (let column = 1; column < originalColumns; column += 1) {
    sheet.getRange(`${columnName(column)}:${columnName(column)}`).format.columnWidth = 16;
  }
  sheet.getRange(`${reviewStart}:${reviewStart}`).format.columnWidth = 18;
  sheet.getRange(`${columnName(originalColumns + 1)}:${columnName(originalColumns + 1)}`).format.columnWidth = 14;
  sheet.getRange(`${decisionColumn}:${decisionColumn}`).format.columnWidth = 30;
  sheet.getRange(`${columnName(originalColumns + 3)}:${columnName(originalColumns + 3)}`).format.columnWidth = 48;
  sheet.getRange(`${columnName(originalColumns + 4)}:${columnName(originalColumns + 4)}`).format.columnWidth = 42;
  sheet.getRange(`${followUpColumn}:${followUpColumn}`).format.columnWidth = 17;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
}

async function createWorkbook(atlasDir, outputDir, definition, manifest, sourceRows) {
  const csvPath = path.join(atlasDir, "review_queues", `${definition.key}.csv`);
  const csvText = await fs.readFile(csvPath, "utf8");
  const workbookCsv = selectWorkbookColumns(csvText, definition.workbookColumns);
  const workbook = await Workbook.fromCSV(workbookCsv, { sheetName: "Review Queue" });
  await decorateQueue(workbook, workbookCsv, definition, manifest);
  await addInstructions(workbook, definition, manifest, sourceRows);
  await addSnapshot(workbook, manifest, definition);
  await addVocabulary(workbook, definition);
  await addEvidenceLinks(workbook, sourceRows);
  await workbook.recalculate();
  // Inspect only the compact instructions page. Inspecting the full 100k-row
  // queue would defeat the purpose of streaming it into a workbook.
  const inspection = await workbook.inspect({
    kind: "region",
    sheetId: "Instructions",
    range: "A1:H12",
    maxChars: 1200,
  });
  if (!inspection) {
    throw new Error(`Workbook inspection failed for ${definition.key}`);
  }
  const output = await SpreadsheetFile.exportXlsx(workbook);
  const destination = path.join(outputDir, definition.filename);
  await output.save(destination);
  return destination;
}

async function main() {
  const { atlas, output, queue } = parseArgs(process.argv.slice(2));
  const atlasDir = path.resolve(atlas);
  const outputDir = path.resolve(output);
  const manifest = JSON.parse(await fs.readFile(path.join(atlasDir, "atlas_manifest.json"), "utf8"));
  const sourceRows = JSON.parse(await fs.readFile(path.join(atlasDir, "site", "assets", "data", "sources.json"), "utf8"));
  await fs.mkdir(outputDir, { recursive: true });
  const definitions = queue
    ? queueDefinitions.filter((definition) => definition.key === queue)
    : queueDefinitions;
  if (!definitions.length) throw new Error(`Unknown review queue: ${queue}`);
  const written = [];
  for (const definition of definitions) {
    written.push(await createWorkbook(atlasDir, outputDir, definition, manifest, sourceRows));
  }
  process.stdout.write(`${JSON.stringify({ written }, null, 2)}\n`);
}

await main();
