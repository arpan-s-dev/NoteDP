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
