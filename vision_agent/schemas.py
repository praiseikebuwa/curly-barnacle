from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

def utc_now() -> datetime: return datetime.now(timezone.utc)
class BoundingBox(BaseModel):
    """Pixel coordinates, origin top-left, absolute image/screen pixels."""
    x: int = Field(ge=0); y: int = Field(ge=0); width: int = Field(ge=0); height: int = Field(ge=0)
class InterfaceElement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    element_id: str; observation_id: str | None = None; element_type: str = "unknown"
    accessible_role: str | None = None; visible_label: str | None = None; accessible_name: str | None = None
    bounding_box: BoundingBox | None = None; text_content: str | None = None
    enabled_state: bool | None = None; visible_state: bool | None = None; selected_state: bool | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    detection_source: Literal["ocr", "uia", "browser_dom", "object_detector", "vlm", "human"] = "ocr"
    application_context: str | None = None; parent_element_id: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict); observed_at: datetime = Field(default_factory=utc_now)
class ActionType(str, Enum):
    capture_screen="capture_screen"; click_element="click_element"; click_coordinates="click_coordinates"; double_click_element="double_click_element"; move_pointer="move_pointer"; type_text="type_text"; press_key="press_key"; scroll="scroll"; wait="wait"; open_authorized_url="open_authorized_url"; request_user_approval="request_user_approval"; finish_task="finish_task"
class ActionRequest(BaseModel):
    action_id: str; action_type: ActionType; target_reference: str | None = None; parameters: dict[str, Any] = Field(default_factory=dict)
    intended_outcome: str = ""; risk_classification: Literal["low", "medium", "high", "irreversible"] = "low"; requires_confirmation: bool = False; verification_strategy: str = "screen_change"
    @field_validator("parameters")
    @classmethod
    def reject_code(cls, value):
        if {"shell","command","python","code","exec"}.intersection(value): raise ValueError("arbitrary code or shell parameters are not allowed")
        return value
class VerificationStatus(str, Enum):
    verified_success="VERIFIED_SUCCESS"; verified_failure="VERIFIED_FAILURE"; uncertain="UNCERTAIN"
class VerificationResult(BaseModel):
    expected_condition: str; observed_evidence: str; status: VerificationStatus; method: str; confidence: float | None = Field(default=None, ge=0, le=1); timestamp: datetime = Field(default_factory=utc_now); associated_action: str | None = None
class TaskState(str, Enum):
    created="CREATED"; validating="VALIDATING"; planning="PLANNING"; waiting_for_approval="WAITING_FOR_APPROVAL"; ready="READY"; running="RUNNING"; verifying="VERIFYING"; recovering="RECOVERING"; completed="COMPLETED"; failed="FAILED"; cancelled="CANCELLED"; needs_user_input="NEEDS_USER_INPUT"
class TaskCreate(BaseModel):
    goal: str = Field(min_length=1, max_length=4000); dry_run: bool = True; max_actions: int = Field(default=20, ge=1, le=100); timeout_seconds: int = Field(default=300, ge=1, le=3600)
class TaskRecord(BaseModel):
    task_id: str; goal: str; state: TaskState; created_at: datetime; updated_at: datetime; dry_run: bool; max_actions: int; events: list[dict[str, Any]] = Field(default_factory=list)
class HealthResponse(BaseModel): status: Literal["ok","degraded"]; version: str; capabilities: dict[str, bool]
