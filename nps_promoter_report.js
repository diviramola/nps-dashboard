const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak
} = require("docx");

// Colors
const ACCENT = "2E5090";
const LIGHT_BG = "E8EEF4";
const HEADER_BG = "2E5090";
const WHITE = "FFFFFF";
const GRAY = "666666";
const LIGHT_GRAY = "CCCCCC";

const border = { style: BorderStyle.SINGLE, size: 1, color: LIGHT_GRAY };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0 };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

// Page: US Letter, 1 inch margins => content width 9360 DXA
const PAGE_W = 12240;
const PAGE_H = 15840;
const MARGIN = 1440;
const CONTENT_W = PAGE_W - 2 * MARGIN; // 9360

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: HEADER_BG, type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, color: WHITE, font: "Arial", size: 18 })]
    })]
  });
}

function dataCell(text, width, opts = {}) {
  const { bold, align, shading } = opts;
  const cellOpts = {
    borders,
    width: { size: width, type: WidthType.DXA },
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({
      alignment: align || AlignmentType.CENTER,
      children: [new TextRun({ text, bold: !!bold, font: "Arial", size: 18, color: "333333" })]
    })]
  };
  if (shading) cellOpts.shading = { fill: shading, type: ShadingType.CLEAR };
  return cellOpts;
}

function makeTable(headers, rows, colWidths) {
  const tableRows = [
    new TableRow({ children: headers.map((h, i) => headerCell(h, colWidths[i])) }),
    ...rows.map((row, ri) =>
      new TableRow({
        children: row.map((cell, ci) => {
          const opts = { shading: ri % 2 === 1 ? "F5F7FA" : undefined };
          if (ci === 0) opts.align = AlignmentType.LEFT;
          if (typeof cell === "object") {
            return new TableCell(dataCell(cell.text, colWidths[ci], { ...opts, bold: cell.bold }));
          }
          return new TableCell(dataCell(cell, colWidths[ci], opts));
        })
      })
    )
  ];
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: tableRows
  });
}

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 28, color: ACCENT })]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 24, color: ACCENT })]
  });
}

function para(text, opts = {}) {
  const runs = [];
  if (typeof text === "string") {
    runs.push(new TextRun({ text, font: "Arial", size: 20, color: "333333", ...opts }));
  } else {
    text.forEach(t => {
      if (typeof t === "string") runs.push(new TextRun({ text: t, font: "Arial", size: 20, color: "333333" }));
      else runs.push(new TextRun({ font: "Arial", size: 20, color: "333333", ...t }));
    });
  }
  return new Paragraph({
    spacing: { after: 160, line: 276 },
    children: runs
  });
}

function quote(location, text) {
  return new Paragraph({
    spacing: { after: 120, line: 276 },
    indent: { left: 480 },
    border: { left: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 8 } },
    children: [
      new TextRun({ text: `[${location}]  `, font: "Arial", size: 18, bold: true, color: GRAY, italics: true }),
      new TextRun({ text: `\u201C${text}\u201D`, font: "Arial", size: 18, color: GRAY, italics: true })
    ]
  });
}

function bullet(textParts) {
  const runs = [];
  runs.push(new TextRun({ text: "\u2022  ", font: "Arial", size: 20, color: ACCENT, bold: true }));
  if (typeof textParts === "string") {
    runs.push(new TextRun({ text: textParts, font: "Arial", size: 20, color: "333333" }));
  } else {
    textParts.forEach(t => {
      if (typeof t === "string") runs.push(new TextRun({ text: t, font: "Arial", size: 20, color: "333333" }));
      else runs.push(new TextRun({ font: "Arial", size: 20, color: "333333", ...t }));
    });
  }
  return new Paragraph({
    spacing: { after: 100, line: 276 },
    indent: { left: 360 },
    children: runs
  });
}

function spacer(pts = 100) {
  return new Paragraph({ spacing: { after: pts }, children: [] });
}

// --- BUILD DOCUMENT ---

const children = [];

// Title block
children.push(new Paragraph({
  spacing: { after: 80 },
  children: [new TextRun({ text: "NPS PROMOTER ANALYSIS", font: "Arial", size: 36, bold: true, color: ACCENT })]
}));
children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "Sprint 11 \u2192 RSP3  |  Dec 2025 \u2013 Mar 2026", font: "Arial", size: 22, color: GRAY })]
}));
children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "Total respondents: 5,580  |  Promoters: 2,525", font: "Arial", size: 22, color: GRAY })]
}));
children.push(new Paragraph({
  spacing: { after: 240 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 8 } },
  children: []
}));

// Preface
children.push(new Paragraph({
  spacing: { after: 240 },
  shading: { fill: LIGHT_BG, type: ShadingType.CLEAR },
  children: [
    new TextRun({ text: "This report reads the data from the promoter\u2019s lens \u2014 who they are, what\u2019s holding their loyalty, and what they\u2019re telling us (and not telling us).", font: "Arial", size: 20, italics: true, color: ACCENT })
  ]
}));

// Section 1
children.push(heading1("1. What Are Promoters Responding To?"));
children.push(para("The promoter share has moved from 46% to 41% across 7 sprints. Before unpacking what\u2019s driving that shift, it\u2019s worth noting a measurement constraint: 67% of tagged promoters sit under \u201CGeneral Good Service\u201D \u2014 so for most promoters, we don\u2019t actually have a specific reason on record. Only 28% carry a usable tag. What follows is an attempt to read the promoter signal through that narrow window \u2014 the specific tags we do have, the free-text comments, and the service ticket data \u2014 to understand what these customers value, what keeps them loyal, and where that loyalty has conditions attached. S14, the one sprint where promoters crossed 50%, is worth examining closely for what it can tell us about what a good sprint looks like from the promoter\u2019s side."));
children.push(spacer(80));

// NPS table
const npsHeaders = ["Sprint", "N", "Promoter %", "Detractor %", "NPS"];
const npsColWidths = [1800, 1200, 2120, 2120, 2120];
const npsRows = [
  ["S11", "714", "46.1%", "46.8%", "-0.7"],
  ["S12", "691", "46.2%", "47.2%", "-1.0"],
  ["S13", "816", "46.0%", "44.2%", "+1.7"],
  ["S14", "711", "50.2%", "44.7%", "+5.5 \u25B2"],
  ["RSP1", "908", "45.9%", "47.2%", "-1.3"],
  ["RSP2", "800", "42.9%", "49.6%", "-6.8"],
  ["RSP3", "940", "41.0%", "52.3%", "-11.4 \u25BC"],
  [{ text: "Total", bold: true }, { text: "5,580", bold: true }, { text: "45.3%", bold: true }, { text: "47.6%", bold: true }, { text: "-2.4", bold: true }]
];
children.push(makeTable(npsHeaders, npsRows, npsColWidths));
children.push(spacer(160));

// Section 2
children.push(heading1("2. A Note on What We Can and Can\u2019t See"));
children.push(para("The specificity gap between promoters and detractors is worth sitting with. When a detractor is tagged, 58% of the time they receive a specific problem label. When a promoter is tagged, 67% of the time they receive \u201CGeneral Good Service\u201D \u2014 a catch-all that typically corresponds to one- or two-word comments like \u201Cgood service\u201D, \u201Cnice\u201D, or \u201Call okay\u201D \u2014 content that confirms satisfaction but gives no signal on what\u2019s driving it. Detractors also write nearly twice as much in free text (60 chars vs 34 chars). This isn\u2019t a criticism of the tagging process \u2014 it\u2019s natural to describe problems more precisely than satisfaction. But it does mean our understanding of what\u2019s working is structurally thinner than our understanding of what\u2019s broken."));
children.push(spacer(80));

// Tagging table
const tagHeaders = ["Segment", "N", "Has comment", "Tagged", "Generic tag", "Specific tag", "Avg length"];
const tagColWidths = [1300, 900, 1200, 1000, 1320, 1320, 1320];
const tagRows = [
  ["Promoters", "2,525", "50.9%", "28.1%", "67.1%", "28.3%", "34 chars"],
  ["Detractors", "2,657", "68.9%", "40.3%", "35.8%", "58.4%", "60 chars"]
];
children.push(makeTable(tagHeaders, tagRows, tagColWidths));
children.push(spacer(80));
children.push(para("What follows is based on the 28% of tagged promoters with specific tags, plus keyword analysis of free-text comments.", { italics: true, color: GRAY }));
children.push(spacer(160));

// Section 3
children.push(heading1("3. What Promoters Are Responding To"));

// 3a - Affordable
children.push(heading2("Affordable / Value for Money \u2014 119 promoters (4.7%)"));
children.push(para("This is the most consistent signal across all sprints (3\u20137% per sprint; all sprint-level counts below 30 and directional). What\u2019s interesting is how these customers talk about value \u2014 it\u2019s less about being cheap and more about access. They describe a service level a middle-class household can actually afford, with flexible plan sizes and zero installation cost. This group also appears to be the most likely to refer unprompted, which raises the question: is affordability doing more work for acquisition than we\u2019re giving it credit for?"));
children.push(quote("RSP2, Delhi", "Koi bhi person low budget me apna wifi lgwa sakta hai woh bhi apne manpasand plan ke sath... jaise uska budget 500\u20B9 hai toh wo person 50 MBps wala connection lagwa skta hai"));
children.push(quote("S13, Delhi", "I told my dear relatives how cheap and good wifi is, Many of my relatives are enjoying your wifi"));
children.push(quote("RSP3, Delhi", "ise lagwane ki cost zero h aur hume sirf recharge ke paise dene hote h"));
children.push(para([
  { text: "Word-of-mouth signal: ", bold: true },
  "16 promoters explicitly mention recommending to friends or family \u2014 too small to be conclusive (\u26A0 all sprint counts <30), but the clustering within this theme rather than others is worth noting."
]));
children.push(spacer(80));

// 3b - Speed
children.push(heading2("Good Speed \u2014 186 promoters (7.4%)"));
children.push(para("The largest specific positive pool, consistent across sprints (4\u20139% per sprint; directional). What stands out is how promoters talk about speed. They\u2019re not citing Mbps numbers \u2014 they\u2019re describing what speed feels like in use: uninterrupted streaming, multiple devices working at once, a noticeable improvement over JioFiber or a previous connection. The benchmark is experiential, not technical. This suggests that as long as the lived experience holds, these customers stay loyal \u2014 but it also means the loyalty is tied to a feeling that can shift without any change in the underlying specs."));
children.push(quote("RSP1, Delhi", "Mere area mein Wiom 5G bahut hi achha perform kar raha hai. Maine Jio AirFiber 5G bhi use kiya hai, lekin Wiom ki speed zyada fast aur kaafi stable hai"));
children.push(quote("RSP3, Delhi", "speed achi h or speed achi hone se picture quality bhi achi aati h"));
children.push(spacer(80));

// 3c - Complaint Resolution
children.push(heading2("Fast Complaint Resolution \u2014 31 promoters (1.2%)"));
children.push(para("A small group in absolute terms, but something interesting shows up here: customers who had a problem and saw it resolved quickly seem to express stronger advocacy than customers who never had an issue at all. If this holds at scale, there may be a counterintuitive path to promoter creation through service recovery rather than just service avoidance."));
children.push(quote("RSP2, Delhi", "jo bhi problem Hoti Hai Ham contact karte Hain paanch minute mein solve Ho jaati Hai"));
children.push(quote("RSP3, Delhi", "agr kuch bhi problem hoti h to thik bhi jldi ho jata h, vese to jldi problem aati nhi"));
children.push(spacer(80));

// 3d - Support
children.push(heading2("Good Support / Technician \u2014 24 promoters (1.0%)"));
children.push(para("Episodic and too sparse for trends, but the human touchpoints \u2014 call-centre agent tone, technician behaviour during installation \u2014 do register when they go well. Worth asking whether these moments are being left to chance or whether there\u2019s something systematic producing them."));
children.push(spacer(160));

// Section 4
children.push(heading1("4. Where Promoter Loyalty Has Conditions"));
children.push(para("42 promoters (1.7%) rate 9\u201310 but surface a real problem in their tag or comment. A further 10 lead with praise and bury a complaint as a secondary tag. These are worth reading carefully \u2014 they\u2019re telling us what could tip them. Four patterns emerge:"));

children.push(bullet([
  { text: "Reliability tolerance \u2014 ", bold: true },
  "The largest category. These promoters accept intermittent disconnects or slow patches because the price makes it worthwhile. The loyalty is real but conditional: it holds as long as the value equation stays intact. A tariff increase or a credible competitor offering similar value would test this group first."
]));
children.push(bullet([
  { text: "28-day billing \u2014 ", bold: true },
  "A specific, articulate ask for calendar-month plans. It comes up enough to notice, though at current levels it reads more as a wish than a friction point. Worth tracking whether this grows."
]));
children.push(bullet([
  { text: "Range constraint \u2014 ", bold: true },
  "\u201CWiom sabhi jagah acha nhi chalta\u201D \u2014 satisfaction is capped by geography. Not quite a complaint, more of a caveat. These promoters are telling us they\u2019d rate higher if coverage were better."
]));
children.push(bullet([
  { text: "Support roulette \u2014 ", bold: true },
  "One bad technician visit or call-centre interaction sitting inside an otherwise positive relationship. Episodic from what we can see here, not systemic \u2014 but each instance is a reminder that a single human interaction can dent an otherwise strong promoter."
]));

children.push(spacer(80));
children.push(para([
  { text: "Sprint trend on conditional promoters ", bold: true },
  "(\u26A0 all <30): S13: 1.9% \u2192 S14: 0.6% \u2192 RSP1: 3.4% \u2192 RSP2: 1.5% \u2192 RSP3: 3.6%. The RSP3 reading returning to RSP1 levels is the directional flag worth watching \u2014 is this noise, or are more promoters starting to attach caveats?"
]));
children.push(spacer(160));

// Section 5
children.push(heading1("5. What Service Tickets Tell Us About Promoter Creation"));
children.push(para([
  { text: "Tenure \u22653 months. ", bold: true, italics: true, color: GRAY },
  "Ticket types counted: RDNI (Recharge Done No Internet), ISD (Internet Supply Down), Slow Speed / Range, Frequent Disconnection, Partner Misbehavior. Billing, payment, and administrative tickets excluded. Window: Oct 2025 \u2013 Mar 2026."
]));
children.push(spacer(80));

const ticketHeaders = ["Service tickets (3-mo)", "N", "Promoter %", "Detractor %", "NPS"];
const ticketColWidths = [2600, 1200, 1800, 1960, 1800];
const ticketRows = [
  ["0", "1,537", "53.2%", "38.8%", "+14.4"],
  ["1", "1,075", "48.9%", "42.6%", "+6.3"],
  ["2", "783", "44.3%", "49.2%", "-4.9"],
  ["3", "543", "40.5%", "52.5%", "-12.0"],
  ["4+", "1,482", "36.0%", "58.1%", "-22.1"]
];
children.push(makeTable(ticketHeaders, ticketRows, ticketColWidths));
children.push(spacer(80));

children.push(para("The gradient here is steep and consistent. Customers with zero service issues in three months sit at NPS +14.4 \u2014 a strong promoter-majority base. Each additional ticket erodes roughly 9 NPS points. By the time a customer has logged 4+ tickets, promoter share has dropped to 36% and detractors dominate at 58%."));
children.push(para("This raises a question worth exploring: if the clearest path to a promoter is an uninterrupted 3-month experience, how many customers are actually getting that? The 1,482 customers in the 4+ bracket are raising service issues faster than they\u2019re being resolved \u2014 understanding what\u2019s driving repeat tickets for this group could be as important as any acquisition or loyalty programme."));

// Section 6 - Next Steps
children.push(spacer(100));
children.push(heading1("6. Next Steps"));
children.push(para("This analysis opens several threads worth pulling on. The following are areas where additional data could sharpen the picture:"));

children.push(bullet([
  { text: "Missed call rates \u2014 ", bold: true },
  "How many promoters are trying to reach support and not getting through? If the service recovery path (Section 3) is as powerful as it appears, missed calls could be silently converting would-be promoters into passives or detractors. Overlaying missed call data by sprint could help explain part of the S14 \u2192 RSP3 decline."
]));
children.push(bullet([
  { text: "S14 deep dive \u2014 ", bold: true },
  "S14 remains the only sprint with a promoter majority. Was there a network improvement, a seasonal effect, a cohort difference, or a campaign running? Identifying the driver would tell us whether it\u2019s replicable."
]));
children.push(bullet([
  { text: "Repeat ticket root causes \u2014 ", bold: true },
  "The 1,482 customers with 4+ service tickets (NPS -22) are the clearest at-risk group. Breaking down the ticket sequence for this cohort \u2014 what\u2019s the first ticket type, what recurs, and how long between tickets \u2014 could reveal whether this is a systemic infrastructure issue or a resolution-quality issue."
]));
children.push(bullet([
  { text: "Promoter tagging improvement \u2014 ", bold: true },
  "With 67% of promoters tagged generically, we\u2019re working with a partial view. Even a simple secondary prompt (\u201CWhat specifically do you like most?\u201D) for 9\u201310 raters could double the usable signal without changing the survey structure."
]));
children.push(bullet([
  { text: "Affordability \u2192 referral tracking \u2014 ", bold: true },
  "The 16 unprompted referral mentions clustered in the affordability theme are directional, but if referral codes or source tracking exists, cross-referencing actual referral conversions against NPS scores could validate whether promoters are genuinely driving acquisition."
]));

// Source line
children.push(spacer(200));
children.push(new Paragraph({
  border: { top: { style: BorderStyle.SINGLE, size: 2, color: LIGHT_GRAY, space: 8 } },
  spacing: { before: 100 },
  children: [new TextRun({
    text: "Source: NPS Verma Parivar.xlsx (Sprint 11 Dec\u201925 \u2013 RSP3 Mar\u201926) + Metabase service_ticket_model. Sprint-level promoter theme counts are all <30 and should be treated as directional only. Ticket window: Oct 2025 \u2013 Mar 2026 scoped to service ticket types only.",
    font: "Arial", size: 16, color: GRAY, italics: true
  })]
}));

// --- ASSEMBLE ---
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: ACCENT, space: 4 } },
          children: [new TextRun({ text: "NPS Promoter Analysis  |  Sprint 11 \u2192 RSP3", font: "Arial", size: 16, color: GRAY })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          border: { top: { style: BorderStyle.SINGLE, size: 2, color: ACCENT, space: 4 } },
          children: [
            new TextRun({ text: "Page ", font: "Arial", size: 16, color: GRAY }),
            new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: GRAY })
          ]
        })]
      })
    },
    children
  }]
});

const OUTPUT = "C:/Users/divir/claude code/NPS_Promoter_Analysis_S11_RSP3.docx";
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUTPUT, buf);
  console.log("Created: " + OUTPUT);
});
