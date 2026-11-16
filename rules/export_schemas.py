"""규칙 코드 입출력 형식을 `schemas/rules.json` 에 모은다. 실행: python rules/export_schemas.py"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

from contact_priority.schema import FarmIndicator, PrioritizedFarm  # noqa: E402
from deadline.schema import DeadlineRequest, DeadlineResponse  # noqa: E402
from disease_branch.schema import DiseaseBranch, MatchedSymptoms, RequiredPart, SymptomAnswers  # noqa: E402
from report_state_machine.schema import Form6, ReportStateRequest, ReportStateResponse  # noqa: E402
from routine_change_alert.schema import AlertState, DiaryEntry, RoutineChange  # noqa: E402

DOC = {
    "deadline": {"request": DeadlineRequest, "response": DeadlineResponse},
    "report_state": {"request": ReportStateRequest, "response": ReportStateResponse},
    "form6": {"response": Form6},
    "disease_branch": {
        "request": SymptomAnswers,
        "matched_symptoms": MatchedSymptoms,
        "required_parts_item": RequiredPart,
        "disease_branch": DiseaseBranch,
    },
    "routine_change": {"diary_entry": DiaryEntry, "response": RoutineChange, "alert_state": AlertState},
    "contact_priority": {"request_item": FarmIndicator, "response_item": PrioritizedFarm},
}

out = HERE.parents[1] / "schemas" / "rules.json"
data = {name: {k: m.model_json_schema() for k, m in parts.items()} for name, parts in DOC.items()}
out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"저장: {out}")
