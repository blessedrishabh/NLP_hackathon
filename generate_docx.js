/**
 * generate_docx.js
 * =============================================================================
 * Generates Method_Statement_RCC.docx from generated_sections_debug.json
 *
 * Usage:
 *   node generate_docx.js \
 *     --sections_json output/generated_sections_debug.json \
 *     --out_path      output/Method_Statement_RCC.docx \
 *     --team_name     "musketeers"
 */

const fs   = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, LevelFormat, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, NumberFormat,
  PageBreak, TabStopType, TabStopPosition,
} = require("docx");

// ══════════════════════════════════════════════════════════════════════════
// CLI ARGS
// ══════════════════════════════════════════════════════════════════════════
const args = process.argv.slice(2);
const getArg = (flag, fallback) => {
  const i = args.indexOf(flag);
  return i !== -1 && args[i + 1] ? args[i + 1] : fallback;
};

const SECTIONS_JSON = getArg("--sections_json", "output/generated_sections_debug.json");
const OUT_PATH      = getArg("--out_path",      "output/Method_Statement_RCC.docx");
const TEAM_NAME     = getArg("--team_name",     "Musketeers");
const TEAM_MEMBERS = [
    "Rishabh Sharma (Team Leader)",
    "Aman Likhitkar",
    "Rishabh Bharadwaj",
    "Deepak Pachauri",
    ];

// ══════════════════════════════════════════════════════════════════════════
// SECTION DEFINITIONS
// ══════════════════════════════════════════════════════════════════════════
const MS_SECTIONS = [
  { key: "purpose",       heading: "1.  Purpose of the Method Statement"             },
  { key: "scope",         heading: "2.  Scope of the Method Statement"               },
  { key: "acronyms",      heading: "3.  Acronyms and Definitions"                    },
  { key: "references",    heading: "4.  Reference Documents"                         },
  { key: "procedure",     heading: "5.  Procedure for Concreting"                    },
  { key: "equipment",     heading: "6.  Equipment Used"                              },
  { key: "personnel",     heading: "7.  Key People Involved"                         },
  { key: "quality",       heading: "8.  Quality Control and Testing"                 },
  { key: "health_safety", heading: "9.  Health, Safety & Environment (HSE)"          },
  { key: "other",         heading: "10. Other Relevant Information"                  },
];

// ══════════════════════════════════════════════════════════════════════════
// DESIGN TOKENS
// ══════════════════════════════════════════════════════════════════════════
const COLOR = {
  brand:       "1F4E79",   // deep navy  — primary brand
  accent:      "2E75B6",   // mid-blue   — heading underlines, table header
  accentLight: "D6E4F0",   // pale blue  — table header fill
  bodyText:    "1A1A1A",   // near-black — body copy
  mutedText:   "595959",   // grey       — meta / captions
  white:       "FFFFFF",
  ruleLine:    "BDD7EE",   // light blue — divider rules
};

const FONT = { body: "Calibri", heading: "Calibri" };

// DXA helpers (1 inch = 1440 DXA)
const IN  = (n) => Math.round(n * 1440);
const PT  = (n) => n * 2;          // half-points in docx
const CM  = (n) => Math.round(n * 567);

// Page geometry (A4)
const PAGE_W       = 11906;
const PAGE_H       = 16838;
const MARGIN_TOP   = IN(0.75);
const MARGIN_BOT   = IN(0.75);
const MARGIN_LEFT  = IN(1.0);
const MARGIN_RIGHT = IN(1.0);
const CONTENT_W    = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT;  // 9026 DXA

// ══════════════════════════════════════════════════════════════════════════
// HELPER BUILDERS
// ══════════════════════════════════════════════════════════════════════════

/** Thin horizontal rule via paragraph bottom border */
function hRule(color = COLOR.ruleLine, thickness = 6) {
  return new Paragraph({
    spacing: { before: 0, after: 0 },
    border: {
      bottom: { style: BorderStyle.SINGLE, size: thickness, color, space: 1 },
    },
    children: [],
  });
}

/** Empty spacer paragraph */
function spacer(before = 40, after = 40) {
  return new Paragraph({ spacing: { before, after }, children: [] });
}

/** Plain body paragraph */
function bodyPara(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 20, after: 40, line: 252, lineRule: "auto" },
    alignment: opts.alignment || AlignmentType.JUSTIFIED,
    children: [
      new TextRun({
        text,
        font: FONT.body,
        size: PT(11),
        color: COLOR.bodyText,
        bold: opts.bold || false,
      }),
    ],
  });
}

/** Bold-key inline paragraph: "KEY — value" */
function keyValuePara(key, value) {
  return new Paragraph({
    spacing: { before: 20, after: 30, line: 252, lineRule: "auto" },
    children: [
      new TextRun({ text: key, font: FONT.body, size: PT(11), bold: true,  color: COLOR.brand }),
      new TextRun({ text: `  —  ${value}`, font: FONT.body, size: PT(11), color: COLOR.bodyText }),
    ],
  });
}

/** Bullet list item */
function bulletItem(text, indent = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level: indent },
    spacing: { before: 20, after: 30, line: 240, lineRule: "auto" },
    children: [
      new TextRun({ text, font: FONT.body, size: PT(11), color: COLOR.bodyText }),
    ],
  });
}

/** Numbered list item */
function numberedItem(text, indent = 0) {
  return new Paragraph({
    numbering: { reference: "numbers", level: indent },
    spacing: { before: 20, after: 30, line: 240, lineRule: "auto" },
    children: [
      new TextRun({ text, font: FONT.body, size: PT(11), color: COLOR.bodyText }),
    ],
  });
}

/** Section heading (Heading 1 style with brand color + underline rule) */
function sectionHeading(text) {
  return [
    spacer(40, 0),
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      spacing: { before: 0, after: 30 },
      children: [
        new TextRun({
          text,
          font: FONT.heading,
          size: PT(13),
          bold: true,
          color: COLOR.brand,
          allCaps: false,
        }),
      ],
    }),
    hRule(COLOR.accent, 6),
    spacer(20, 0),
  ];
}

/** Sub-heading (Heading 2 style) */
function subHeading(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 60, after: 30 },
    children: [
      new TextRun({
        text,
        font: FONT.heading,
        size: PT(12),
        bold: true,
        color: COLOR.accent,
      }),
    ],
  });
}

// ══════════════════════════════════════════════════════════════════════════
// CONTENT PARSER
// Converts raw LLM text into styled docx paragraphs
// ══════════════════════════════════════════════════════════════════════════
function parseContent(raw) {
  const lines   = raw.split("\n");
  const result  = [];
  const BULLET  = /^[-*•]\s+/;
  const NUMBERED = /^\d+[\.\)]\s+/;
  const BOLD_LINE = /^\*\*(.*?)\*\*\s*$/;
  const INLINE_BOLD = /\*\*(.*?)\*\*/g;

  // Convert inline **bold** markers into mixed TextRun array
  function richRuns(text) {
    const runs = [];
    let last = 0;
    let m;
    INLINE_BOLD.lastIndex = 0;
    while ((m = INLINE_BOLD.exec(text)) !== null) {
      if (m.index > last) {
        runs.push(new TextRun({ text: text.slice(last, m.index), font: FONT.body, size: PT(11), color: COLOR.bodyText }));
      }
      runs.push(new TextRun({ text: m[1], font: FONT.body, size: PT(11), bold: true, color: COLOR.bodyText }));
      last = INLINE_BOLD.lastIndex;
    }
    if (last < text.length) {
      runs.push(new TextRun({ text: text.slice(last), font: FONT.body, size: PT(11), color: COLOR.bodyText }));
    }
    return runs.length ? runs : [new TextRun({ text, font: FONT.body, size: PT(11), color: COLOR.bodyText })];
  }

  for (const raw_line of lines) {
    const line = raw_line.trimEnd();
    if (!line.trim()) {
      result.push(spacer(0, 0));
      continue;
    }

    // Full-line bold heading e.g. **Formwork:**
    if (BOLD_LINE.test(line)) {
      const headText = line.replace(/^\*\*|\*\*\s*$/g, "");
      result.push(subHeading(headText));
      continue;
    }

    // Bullet
    if (BULLET.test(line.trim())) {
      const text = line.trim().replace(BULLET, "");
      result.push(new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        spacing: { before: 20, after: 30, line: 240, lineRule: "auto" },
        children: richRuns(text),
      }));
      continue;
    }

    // Numbered
    if (NUMBERED.test(line.trim())) {
      const text = line.trim().replace(NUMBERED, "");
      result.push(new Paragraph({
        numbering: { reference: "numbers", level: 0 },
        spacing: { before: 20, after: 30, line: 240, lineRule: "auto" },
        children: richRuns(text),
      }));
      continue;
    }

    // Plain paragraph with inline bold support
    result.push(new Paragraph({
      spacing: { before: 20, after: 40, line: 252, lineRule: "auto" },
      alignment: AlignmentType.JUSTIFIED,
      children: richRuns(line.trim()),
    }));
  }

  return result;
}

// ══════════════════════════════════════════════════════════════════════════
// COVER PAGE
// ══════════════════════════════════════════════════════════════════════════
function buildCoverPage(teamName) {
  const today = new Date().toLocaleDateString("en-IN", {
    day: "2-digit", month: "long", year: "numeric"
  });
  

  const cellBorder = { style: BorderStyle.NIL, size: 0, color: "FFFFFF" };
  const noBorders  = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };

  return [
    // Top colour band via shaded table row
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA },
      columnWidths: [CONTENT_W],
      rows: [
        new TableRow({
          children: [
            new TableCell({
              borders: noBorders,
              shading: { fill: COLOR.brand, type: ShadingType.CLEAR },
              margins: { top: IN(0.3), bottom: IN(0.3), left: IN(0.2), right: IN(0.2) },
              width: { size: CONTENT_W, type: WidthType.DXA },
              children: [
                new Paragraph({
                  alignment: AlignmentType.CENTER,
                  spacing: { before: 0, after: 0 },
                  children: [
                    new TextRun({
                      text: "METHOD STATEMENT",
                      font: FONT.heading, size: PT(28), bold: true, color: COLOR.white,
                    }),
                  ],
                }),
                new Paragraph({
                  alignment: AlignmentType.CENTER,
                  spacing: { before: 60, after: 0 },
                  children: [
                    new TextRun({
                      text: "Reinforced Cement Concrete (RCC) Works",
                      font: FONT.heading, size: PT(16), color: COLOR.accentLight,
                    }),
                  ],
                }),
              ],
            }),
          ],
        }),
      ],
    }),

    spacer(IN(0.15), 0),
    hRule(COLOR.accent, 10),
    spacer(IN(0.15), 0),

    // Meta block
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA },
      columnWidths: [CM(4), CONTENT_W - CM(4)],
      rows: [
        _metaRow("Prepared by:", teamName),
        _metaRowMulti("Team Members:", TEAM_MEMBERS),
        _metaRow("Date:", today),
        _metaRow("Document Ref.:", "Based on CPWD Prescriptive Specifications"),
        ],
    }),

    spacer(IN(0.2), 0),
    hRule(COLOR.ruleLine, 4),
    spacer(IN(0.1), 0),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 0 },
      children: [
        new TextRun({
          text: "CONFIDENTIAL — FOR PROJECT USE ONLY",
          font: FONT.body, size: PT(9), color: COLOR.mutedText,
          allCaps: true, italics: true,
        }),
      ],
    }),

    // Page break to content
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

function _metaRow(label, value) {
  const cellBorder = { style: BorderStyle.NIL, size: 0, color: "FFFFFF" };
  const noBorders  = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };
  const COL1 = CM(4);
  const COL2 = CONTENT_W - CM(4);
  return new TableRow({
    children: [
      new TableCell({
        borders: noBorders,
        width: { size: COL1, type: WidthType.DXA },
        margins: { top: 80, bottom: 80, left: 0, right: 120 },
        children: [new Paragraph({
          children: [new TextRun({ text: label, font: FONT.body, size: PT(11), bold: true, color: COLOR.brand })],
        })],
      }),
      new TableCell({
        borders: noBorders,
        width: { size: COL2, type: WidthType.DXA },
        margins: { top: 80, bottom: 80, left: 120, right: 0 },
        children: [new Paragraph({
          children: [new TextRun({ text: value, font: FONT.body, size: PT(11), color: COLOR.bodyText })],
        })],
      }),
    ],
  });
}
function _metaRowMulti(label, values) {
  const cellBorder = { style: BorderStyle.NIL, size: 0, color: "FFFFFF" };
  const noBorders  = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };

  const COL1 = CM(4);
  const COL2 = CONTENT_W - CM(4);

  return new TableRow({
    children: [
      new TableCell({
        borders: noBorders,
        width: { size: COL1, type: WidthType.DXA },
        margins: { top: 80, bottom: 80, left: 0, right: 120 },
        children: [
          new Paragraph({
            children: [
              new TextRun({
                text: label,
                font: FONT.body,
                size: PT(11),
                bold: true,
                color: COLOR.brand
              })
            ]
          })
        ]
      }),
      new TableCell({
        borders: noBorders,
        width: { size: COL2, type: WidthType.DXA },
        margins: { top: 80, bottom: 80, left: 120, right: 0 },
        children: values.map(name =>
          new Paragraph({
            children: [
              new TextRun({
                text: name,
                font: FONT.body,
                size: PT(11),
                color: COLOR.bodyText
              })
            ]
          })
        )
      })
    ]
  });
}

// ══════════════════════════════════════════════════════════════════════════
// NOTES PAGE (trailing)
// ══════════════════════════════════════════════════════════════════════════
function buildNotesPage() {
  return [
    ...sectionHeading("Document Conventions"),
    bodyPara(
      "Throughout this document, the symbol '§' (section sign) refers to specific ",
      "clauses or subsections within the CPWD Prescriptive Specifications. ",
      "For example, '§5.1.3' indicates Clause 5.1.3 of the specification document."
    ),
    spacer(30, 0),

    ...sectionHeading("Note on Sources"),
    bodyPara(
      "All information in this Method Statement has been extracted from the " +
      "CPWD Prescriptive Specifications document using an NLP-based retrieval " +
      "pipeline (parser v3 + embed_and_store v4, retrieve_and_generate v6).  " +
      "Where specification text was insufficient, this has been explicitly noted " +
      "within the relevant section."
    ),
    spacer(30, 0),
    hRule(COLOR.ruleLine, 4),
    spacer(20, 0),
    new Paragraph({
      spacing: { before: 0, after: 0 },
      children: [
        new TextRun({
          text: "Pipeline parameters:",
          font: FONT.body, size: PT(10), bold: true, color: COLOR.mutedText,
        }),
      ],
    }),
    keyValuePara("Embedding model",  "all-MiniLM-L12-v2"),
    keyValuePara("LLM",              "llama-3.3-70b-versatile (Groq)"),
    keyValuePara("BERT eval model",  "roberta-large (rescaled baseline)"),
    keyValuePara("Retrieval depth",  "20 chunks / section (cross-encoder reranked)"),
    keyValuePara("Context window",   "20 000 characters"),
    keyValuePara("Temperature",      "0.1"),
  ];
}

// ══════════════════════════════════════════════════════════════════════════
// HEADER & FOOTER
// ══════════════════════════════════════════════════════════════════════════
function buildHeader(teamName) {
  return new Header({
    children: [
      new Paragraph({
        tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }],
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: COLOR.accent, space: 1 } },
        spacing: { before: 0, after: 80 },
        children: [
          new TextRun({ text: "METHOD STATEMENT — RCC Works", font: FONT.body, size: PT(9), bold: true, color: COLOR.brand }),
          new TextRun({ text: "\t", font: FONT.body }),
          new TextRun({ text: `Prepared by: ${teamName}`, font: FONT.body, size: PT(9), color: COLOR.mutedText }),
        ],
      }),
    ],
  });
}

function buildFooter() {
  return new Footer({
    children: [
      new Paragraph({
        tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }],
        border: { top: { style: BorderStyle.SINGLE, size: 4, color: COLOR.ruleLine, space: 1 } },
        spacing: { before: 80, after: 0 },
        children: [
          new TextRun({ text: "CPWD Specifications — Confidential", font: FONT.body, size: PT(9), color: COLOR.mutedText }),
          new TextRun({ text: "\t", font: FONT.body }),
          new TextRun({ children: ["Page ", PageNumber.CURRENT], font: FONT.body, size: PT(9), color: COLOR.mutedText }),
        ],
      }),
    ],
  });
}

// ══════════════════════════════════════════════════════════════════════════
// DOCUMENT BUILDER
// ══════════════════════════════════════════════════════════════════════════
function buildDocument(sectionsContent, teamName) {
  const children = [];

  // Cover page (no header/footer on this section — handled via separate section if needed)
  children.push(...buildCoverPage(teamName));

  // Body sections
  for (const sec of MS_SECTIONS) {
    const content = sectionsContent[sec.key] || "Content not generated.";
    children.push(...sectionHeading(sec.heading));
    children.push(...parseContent(content));
  }

  // Notes page
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(...buildNotesPage());

  const doc = new Document({
    numbering: {
      config: [
        {
          reference: "bullets",
          levels: [{
            level: 0, format: LevelFormat.BULLET, text: "▸",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 480, hanging: 240 } } },
          }],
        },
        {
          reference: "numbers",
          levels: [{
            level: 0, format: LevelFormat.DECIMAL, text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 480, hanging: 240 } } },
          }],
        },
      ],
    },

    styles: {
      default: {
        document: { run: { font: FONT.body, size: PT(11), color: COLOR.bodyText } },
      },
      paragraphStyles: [
        {
          id: "Heading1", name: "Heading 1",
          basedOn: "Normal", next: "Normal", quickFormat: true,
          run:       { font: FONT.heading, size: PT(13), bold: true, color: COLOR.brand },
          paragraph: { spacing: { before: 100, after: 20 }, outlineLevel: 0 },
        },
        {
          id: "Heading2", name: "Heading 2",
          basedOn: "Normal", next: "Normal", quickFormat: true,
          run:       { font: FONT.heading, size: PT(11), bold: true, color: COLOR.accent },
          paragraph: { spacing: { before: 60, after: 20 }, outlineLevel: 1 },
        },
      ],
    },

    sections: [
      {
        properties: {
          page: {
            size:   { width: PAGE_W, height: PAGE_H },
            margin: { top: MARGIN_TOP, bottom: MARGIN_BOT, left: MARGIN_LEFT, right: MARGIN_RIGHT },
          },
          pageNumberStart: 1,
          pageNumberFormatType: NumberFormat.DECIMAL,
        },
        headers: { default: buildHeader(teamName) },
        footers: { default: buildFooter() },
        children,
      },
    ],
  });

  return doc;
}

// ══════════════════════════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════════════════════════
(async () => {
  if (!fs.existsSync(SECTIONS_JSON)) {
    console.error(`❌  Sections JSON not found: ${SECTIONS_JSON}`);
    process.exit(1);
  }

  console.log(`📖  Loading sections from ${SECTIONS_JSON} …`);
  const sectionsContent = JSON.parse(fs.readFileSync(SECTIONS_JSON, "utf-8"));

  console.log(`🎨  Building document …`);
  const doc = buildDocument(sectionsContent, TEAM_NAME);

  const outDir = path.dirname(OUT_PATH);
  if (outDir && outDir !== ".") fs.mkdirSync(outDir, { recursive: true });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(OUT_PATH, buffer);
  console.log(`✅  Word document saved → ${OUT_PATH}`);
})();