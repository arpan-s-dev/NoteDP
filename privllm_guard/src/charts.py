"""Fictional EHR charts for the local demo.

These are NOT real patients. Names, MRNs, phones, and addresses are invented.
The notes are written in the style of discharge summaries / H&Ps so the
internship UI looks like a chart review, without using identifiable PHI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SyntheticChart:
    id: str
    display_name: str
    mrn: str
    age: int
    sex: str
    specialty: str
    note_type: str
    facility: str
    encounter_date: str
    risk_tag: str  # low | medium | high
    one_line: str
    note: str
    excerpt: str  # short window the tiny DP model can consume


FACILITY = "Fictional Memorial Hospital (SYNTHETIC DATA — not a real patient)"


CHARTS: list[SyntheticChart] = [
    SyntheticChart(
        id="syn-4401",
        display_name="Elena Voss",
        mrn="SYN-4401",
        age=67,
        sex="F",
        specialty="Cardiology",
        note_type="Discharge summary",
        facility=FACILITY,
        encounter_date="March 8, 2026",
        risk_tag="medium",
        one_line="HFrEF with recent decompensation; GDMT titration.",
        excerpt=(
            "Sixty-seven year old woman with reduced ejection fraction admitted for "
            "decompensated heart failure. Echo showed left ventricular hypokinesis. "
            "Started metoprolol and lisinopril. Follow up in cardiology clinic."
        ),
        note="""SYNTHETIC CHART — not a real person.
Fictional Memorial Hospital  |  MRN SYN-4401  |  Encounter 2026-03-08

DISCHARGE SUMMARY — Cardiology
Patient: Elena Voss  |  67-year-old woman  |  Attending: Dr. Priya Nandakumar (fictional)

CHIEF COMPLAINT
Progressive dyspnea on exertion and two-pillow orthopnea for 10 days.

HISTORY OF PRESENT ILLNESS
Ms. Voss is a 67-year-old retired librarian living in Cedar Hollow (fictional township)
who presented with worsening shortness of breath, 4 kg weight gain, and bilateral
leg swelling. She has a known history of HFrEF (EF 32% in 2025) after a non-ST
elevation myocardial infarction. She ran out of furosemide two weeks ago. No chest
pain, no syncope. She was admitted to the step-down unit for IV diuresis.

PAST MEDICAL HISTORY
Heart failure with reduced ejection fraction; hypertension; type 2 diabetes;
stage 3 chronic kidney disease; osteoarthritis.

HOME MEDICATIONS
Metoprolol succinate 50 mg daily; lisinopril 10 mg daily; furosemide 40 mg daily;
metformin 1000 mg twice daily; atorvastatin 40 mg nightly.

HOSPITAL COURSE
Intravenous furosemide produced 3.8 L net negative balance. Repeat echo: LVEF 30–35%
with global hypokinesis, no tamponade. Troponin remained negative after arrival.
Creatinine peaked at 1.6 mg/dL and improved to 1.3. GDMT continued. Diabetes was
managed with a correctional scale; metformin held while GFR was reduced.

DISPOSITION
Discharged home with daughter ( Mira Voss, fictional ). Follow-up cardiology clinic
March 22, 2026. Call 555-014-8802 (fictional) for nurse line.

ASSESSMENT / PLAN
1. Acute on chronic HFrEF — euvolemic; resume furosemide 40 mg; daily weights.
2. HTN / afterload — lisinopril 10 mg; BMP in 1 week.
3. DM2 — restart metformin if creatinine <1.5 at follow-up.
""",
    ),
    SyntheticChart(
        id="syn-4402",
        display_name="Marcus Hale",
        mrn="SYN-4402",
        age=41,
        sex="M",
        specialty="Oncology",
        note_type="Progress note",
        facility=FACILITY,
        encounter_date="January 14, 2026",
        risk_tag="medium",
        one_line="HER2-positive breast cancer on adjuvant trastuzumab.",
        excerpt=(
            "Forty-one year old man with HER2-positive breast cancer completed adjuvant "
            "trastuzumab. Staging remains T2N1M0. Plan continue targeted therapy and "
            "interval imaging."
        ),
        note="""SYNTHETIC CHART — not a real person.
Fictional Memorial Hospital  |  MRN SYN-4402  |  Encounter 2026-01-14

ONCOLOGY PROGRESS NOTE
Patient: Marcus Hale  |  41-year-old man  |  Clinic: Breast medical oncology

INTERVAL HISTORY
Mr. Hale returns after cycle 12 of adjuvant trastuzumab for HER2-positive invasive
ductal carcinoma of the left breast, staged T2N1M0 after lumpectomy and sentinel
node biopsy in July 2025. He reports only grade 1 fatigue. No dyspnea, no palpitations.
He works as a high-school chemistry teacher in Lake Meridian (fictional).

TREATMENT TO DATE
AC-T completed. Trastuzumab 6 mg/kg IV every 3 weeks, currently cycle 12/18.
Radiation completed October 2025. Echo last month: LVEF 58%, stable.

EXAM
Well appearing. Port site clean. No peripheral edema. Heart regular.

ASSESSMENT / PLAN
HER2-positive early breast cancer, adjuvant phase. Continue trastuzumab.
MUGA or echo prior to cycle 15. Return in 3 weeks. Patient phone 555-019-4410 (fictional).
""",
    ),
    SyntheticChart(
        id="syn-4403",
        display_name="Ruth Okonkwo",
        mrn="SYN-4403",
        age=74,
        sex="F",
        specialty="Neurology",
        note_type="Admission H&P",
        facility=FACILITY,
        encounter_date="February 2, 2026",
        risk_tag="medium",
        one_line="Acute right MCA ischemic stroke; tPA considered.",
        excerpt=(
            "Seventy-four year old woman presented with acute right-sided weakness. "
            "Imaging was consistent with ischemic stroke in the MCA territory. "
            "Started aspirin and high-intensity statin."
        ),
        note="""SYNTHETIC CHART — not a real person.
Fictional Memorial Hospital  |  MRN SYN-4403  |  Encounter 2026-02-02  08:41

NEUROLOGY ADMISSION HISTORY AND PHYSICAL
Patient: Ruth Okonkwo  |  74-year-old woman  |  Source: daughter at bedside

HPI
Last known well 06:55. Family noted right face droop, right arm drift, and
garbled speech. EMS to Fictional Memorial. NIHSS 8 on arrival. CT head without
hemorrhage. CTA: left M2 occlusion. Time window exceeded for thrombolysis
(door-to-needle discussion documented). Taken for thrombectomy consideration;
clot retrieved with TICI 2b. Post-procedure NIHSS 4.

PMH
Atrial fibrillation (not on anticoagulation after prior GI bleed), hypertension,
prior TIA 2023.

MEDS
Amlodipine 5 mg; pantoprazole 40 mg; previously warfarin (held).

PLAN
Admit stroke unit. Aspirin 81 mg. Atorvastatin 80 mg. Swallow study. PT/OT.
Restart anticoagulation discussion with GI in 7–14 days. Address: 18 Birch Lane,
North Quay (fictional).
""",
    ),
    SyntheticChart(
        id="syn-4404",
        display_name="Jonah Peck",
        mrn="SYN-4404",
        age=58,
        sex="M",
        specialty="Emergency medicine",
        note_type="ED note",
        facility=FACILITY,
        encounter_date="March 3, 2026",
        risk_tag="medium",
        one_line="Hypertensive urgency with diabetes; ED observation.",
        excerpt=(
            "Fifty-eight year old man arrived with hypertensive urgency and known diabetes. "
            "Systolic blood pressure was severely elevated. Given labetalol. "
            "Discharged with lisinopril and close follow-up."
        ),
        note="""SYNTHETIC CHART — not a real person.
Fictional Memorial Hospital Emergency Department  |  MRN SYN-4404  |  2026-03-03  21:18

ED PROVIDER NOTE
Patient: Jonah Peck  |  58-year-old man  |  Occupation: long-haul truck driver

ARRIVAL
BP 214/118, HR 92, glucose 268. Reports headache 8/10 and blurred vision after
missing antihypertensives during a 14-hour drive. No chest pain, no focal weakness.
History of type 2 diabetes and hypertension. Lives in a motel near Route 7
(fictional) between runs.

COURSE
IV labetalol 10 mg ×2. BP 168/94. ECG: LVH, no STEMI. Troponin negative ×1.
Creatinine 1.4 (baseline 1.1). Head CT negative. Given lisinopril 20 mg PO.
Counseled on medication adherence. Follow-up PCP within 48 hours. Work note
provided. Phone 555-017-2209 (fictional).

DISPOSITION
Discharged, ambulatory. Return precautions for chest pain, confusion, or BP symptoms.
""",
    ),
    SyntheticChart(
        id="syn-4405",
        display_name="Amina Qureshi",
        mrn="SYN-4405",
        age=62,
        sex="F",
        specialty="Internal medicine",
        note_type="Inpatient progress note",
        facility=FACILITY,
        encounter_date="April 11, 2026",
        risk_tag="low",
        one_line="CAP improving on ceftriaxone and azithromycin.",
        excerpt=(
            "Sixty-two year old woman admitted for community-acquired pneumonia. "
            "Treated with ceftriaxone and azithromycin. Hypoxia improved on hospital day three."
        ),
        note="""SYNTHETIC CHART — not a real person.
Fictional Memorial Hospital  |  MRN SYN-4405  |  HD3  |  2026-04-11

INTERNAL MEDICINE PROGRESS NOTE
Patient: Amina Qureshi  |  62-year-old woman

INTERVAL
Community-acquired pneumonia, day 3. Cough productive of yellow sputum, improving.
O2 weaned from 4 L NC to room air this morning. SpO2 94% ambulating. Appetite fair.
No diarrhea on ceftriaxone + azithromycin.

EXAM
T 37.2, HR 88, RR 18, BP 128/76. Crackles right base, decreased. No edema.

DATA
WBC 9.8 ↓ from 14.1. CXR: improving right lower lobe infiltrate. Blood cultures NGTD.

PLAN
Complete 5-day total antibiotics. PT for deconditioning. Discharge tomorrow if
room-air stable. Lives with spouse in Eastbridge Apartments, unit 4B (fictional).
""",
    ),
    SyntheticChart(
        id="syn-4406",
        display_name="Theo Brandt",
        mrn="SYN-4406",
        age=55,
        sex="M",
        specialty="Radiology",
        note_type="CT report",
        facility=FACILITY,
        encounter_date="May 19, 2026",
        risk_tag="low",
        one_line="1 cm RUL nodule; follow-up CT in six months.",
        excerpt=(
            "CT chest demonstrates a 1 cm right-upper-lobe nodule without lymphadenopathy. "
            "Recommend follow-up CT in six months."
        ),
        note="""SYNTHETIC CHART — not a real person.
Fictional Memorial Hospital  |  MRN SYN-4406  |  2026-05-19

CT CHEST WITHOUT CONTRAST
Patient: Theo Brandt  |  55-year-old man  |  Indication: 30 pack-year smoking history,
now quit; cough.

FINDINGS
1. 11 mm solid nodule in the right upper lobe, posterior segment, no fat or
   calcification. No other nodules >4 mm.
2. No mediastinal or hilar lymphadenopathy.
3. No effusion. Mild centrilobular emphysema.

IMPRESSION
Indeterminate 11 mm RUL nodule. Recommend follow-up CT chest in 6 months
(Fleischner-style). Correlate with pulmonary clinic. Ordering PCP: Dr. L. Cho
(fictional). Patient notified via portal.
""",
    ),
    SyntheticChart(
        id="syn-4407",
        display_name="Sofia Alvarez",
        mrn="SYN-4407",
        age=49,
        sex="F",
        specialty="Pathology",
        note_type="Surgical pathology",
        facility=FACILITY,
        encounter_date="June 4, 2026",
        risk_tag="low",
        one_line="Lung biopsy: moderately differentiated adenocarcinoma.",
        excerpt=(
            "Biopsy shows moderately differentiated adenocarcinoma. Immunostains support "
            "a pulmonary origin. Correlate with imaging."
        ),
        note="""SYNTHETIC CHART — not a real person.
Fictional Memorial Hospital  |  MRN SYN-4407  |  Surgical pathology  |  2026-06-04

SPECIMEN
Right upper lobe core biopsy — Sofia Alvarez, 49-year-old woman.

MICROSCOPIC
Cores of lung with an infiltrative glandular proliferation, moderate nuclear
atypia, and desmoplasia. No in situ carcinoma seen on these cores.

IMMUNO
TTF-1 positive, Napsin A positive, p40 negative, CK7 positive, CK20 negative.

DIAGNOSIS
Moderately differentiated adenocarcinoma, consistent with pulmonary origin.
Correlate with CT (see SYN-linked imaging in this fictional system). Molecular
testing (EGFR/ALK/ROS1/KRAS/BRAF/MET/RET/NTRK/PD-L1) recommended on remaining block.
""",
    ),
    SyntheticChart(
        id="syn-4408",
        display_name="Harlan Quill",
        mrn="SYN-4408",
        age=92,
        sex="M",
        specialty="Medical genetics / neurology",
        note_type="Consult",
        facility=FACILITY,
        encounter_date="March 1, 2026",
        risk_tag="high",
        one_line="High re-identification risk: age + rare occupation + tiny town + rare disease.",
        excerpt=(
            "Ninety-two year old retired neurosurgeon from a very small town with a rare "
            "genetic disorder. No name is required for this combination to be identifying."
        ),
        note="""SYNTHETIC CHART — not a real person.
THIS NOTE IS THE PAPER'S QUASI-IDENTIFIER EXAMPLE, written as a full chart.
Fictional Memorial Hospital  |  MRN SYN-4408  |  2026-03-01

GENETICS / NEUROLOGY CONSULT
Patient: Harlan Quill  |  92-year-old man

REASON FOR CONSULT
Metabolic derangement and chronic neuropathy. Referring team asked whether a rare
inborn error of metabolism explains adult presentations.

HISTORY
Mr. Quill is a retired neurosurgeon. He still lives independently in Graymere
Crossing, population ~900 (fictional). Family describes a maple-like odor during
illness in childhood, never formally diagnosed. Adult history of intermittent
encephalopathy with protein loads. No living first-degree relatives in the county.

This combination — very advanced age, highly unique occupation, tiny geography,
and a rare metabolic phenotype — contains no legal name in the assessment below
and is still high risk for re-identification (see paper § qualitative cases).

ASSESSMENT
Possible late-diagnosed maple syrup urine disease variant versus other branched-chain
aminoacidopathy. Do not over-generalize to "dietary irregularities" if confirmatory
labs are pending (paper failure case on rare disease wording).

PLAN
Plasma amino acids, urine organic acids. Dietary protein review with nutrition.
Avoid publishing this constellation in a public model output without generalization
(age band, region not town, occupation class not "neurosurgeon").
""",
    ),
    SyntheticChart(
        id="syn-4409",
        display_name="Leila Nour",
        mrn="SYN-4409",
        age=34,
        sex="F",
        specialty="Obstetrics (postpartum medicine)",
        note_type="Postpartum discharge",
        facility=FACILITY,
        encounter_date="July 9, 2026",
        risk_tag="high",
        one_line="Postpartum preeclampsia; dates and infant sex are quasi-identifiers.",
        excerpt=(
            "Thirty-four year old woman postpartum day two with preeclampsia. "
            "Blood pressure improved on labetalol. Discharged with close follow-up."
        ),
        note="""SYNTHETIC CHART — not a real person.
Fictional Memorial Hospital  |  MRN SYN-4409  |  PPD2  |  2026-07-09

POSTPARTUM DISCHARGE SUMMARY
Patient: Leila Nour  |  34-year-old woman  |  G2P2  |  Delivery 2026-07-07  03:12

COURSE
Primary cesarean for arrest of descent. Infant: term female, 3210 g, Apgars 8 and 9
(fictional). Intrapartum BP 168/102, proteinuria 2+. Started labetalol 200 mg TID.
Magnesium completed 24 hours. Headache resolved. Platelets 142k, Cr 0.7, AST 38.

DISPOSITION
Home with partner. BP cuff provided. Clinic 2026-07-12. Warning symptoms reviewed.
Address on file: 440 Crescent Row, Port Sable (fictional). Email on file:
leila.nour.synth@example.invalid
""",
    ),
    SyntheticChart(
        id="syn-4410",
        display_name="Devon Marsh",
        mrn="SYN-4410",
        age=29,
        sex="M",
        specialty="Psychiatry / IM",
        note_type="Consult",
        facility=FACILITY,
        encounter_date="August 2, 2026",
        risk_tag="high",
        one_line="Occupation + small clinic + exact dates increase linkability.",
        excerpt=(
            "Twenty-nine year old man evaluated for medication-related hyponatremia. "
            "Sodium improved with fluid restriction. Follow up with primary clinic."
        ),
        note="""SYNTHETIC CHART — not a real person.
Fictional Memorial Hospital  |  MRN SYN-4410  |  2026-08-02

MEDICINE CONSULT
Patient: Devon Marsh  |  29-year-old man  |  Occupation: night pharmacist at
the only 24-hour pharmacy in Red Kettle, NM (fictional; pop. 2,100)

HPI
Brought in after a coworker found him confused at shift change 2026-08-01 23:40.
Na 118. On sertraline 150 mg. Polydipsia. CT head negative. Confusion cleared
as Na rose to 128 with 3% saline then fluid restriction.

PLAN
Hold sertraline pending psych. Repeat BMP 2026-08-03. Do not document the
workplace name in any external LLM prompt — it is a singleton quasi-identifier
together with age and town.
""",
    ),
]


def list_charts() -> list[dict]:
    rows = []
    for chart in CHARTS:
        rows.append(
            {
                "id": chart.id,
                "display_name": chart.display_name,
                "mrn": chart.mrn,
                "age": chart.age,
                "sex": chart.sex,
                "specialty": chart.specialty,
                "note_type": chart.note_type,
                "encounter_date": chart.encounter_date,
                "risk_tag": chart.risk_tag,
                "one_line": chart.one_line,
            }
        )
    return rows


def get_chart(chart_id: str) -> SyntheticChart:
    for chart in CHARTS:
        if chart.id == chart_id:
            return chart
    raise KeyError(chart_id)


def all_notes() -> list[str]:
    return [c.note for c in CHARTS] + [c.excerpt for c in CHARTS]
