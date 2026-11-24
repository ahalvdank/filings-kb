"""Pydantic model of the v3 extraction payload (see ``prompts/extract_v3.md``).

The closed vocabularies of the prompt are ``Literal`` types. Attribute
discriminators (``event_type``, ``update_type`` ...) live inside the free-form
``attributes`` dict, so they are checked per entity type by a validator that
reads the allowed values from the same ``Literal`` aliases.

Where the prompt text and the real cached payloads disagreed, the payloads won:
``Risk.category`` is open (127 distinct strings in the sample), so it is not a Literal.
"""

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

EntityType = Literal[
    "Company", "Security", "Person", "Filing",
    "Risk", "LegalProceeding", "GuidanceUpdate", "RevenueSegment",
    "CapitalReturn", "CapitalStructureEvent", "ESGCommitment",
    "MaterialEvent", "InsiderTransaction",
]

RelationshipType = Literal[
    # pre-existing research-report relationships
    "competes_with", "supplies_to", "has_security", "classified_in",
    "has_bull_case", "has_bear_case", "has_catalyst", "has_risk",
    "exposed_to_factor", "performs_in_regime",
    "supports_claim", "contradicts", "applies_to_engine",
    "owns_stock_in", "extracted_from", "filed_by",
    # filing-derived
    "discloses_risk", "discloses_legal_proceeding", "updates_guidance_for",
    "reports_segment", "discloses_event", "discloses_capital_return",
    "discloses_capital_structure_event", "discloses_esg_commitment",
    "transacted_in", "is_officer_of", "risk_heightened_in",
    "risk_resolved_in", "legal_proceeding_status_changed_in",
]

ChunkType = Literal[
    "risk_factor", "legal_proceeding", "guidance", "segment", "capital_return",
    "capital_structure_event", "esg_commitment", "material_event",
    "executive_biography", "insider_holding",
]

LegalStatus = Literal["pending", "settled", "appealed", "dismissed"]
UpdateType = Literal[
    "confirmation", "revision_up", "revision_down", "withdrawal", "initiation",
]
ReturnType = Literal[
    "buyback", "dividend", "special_dividend", "dividend_increase",
    "dividend_cut", "dividend_suspension", "repurchase_authorization",
]
InstrumentType = Literal[
    "notes", "bonds", "term_loan", "revolver", "equity_issuance",
    "equity_repurchase_authorization", "refinancing",
]
CommitmentType = Literal[
    "ghg_reduction", "carbon_neutral", "water_use", "dei",
    "board_composition", "governance_change",
]
EventType = Literal[
    "acquisition", "divestiture", "restructuring", "restatement",
    "accounting_change", "cybersecurity_incident", "management_change",
    "compensation_disclosure", "stock_split", "goodwill_impairment",
    "joint_venture", "regulatory_action", "business_combination",
]
OfficerRole = Literal[
    "CEO", "CFO", "COO", "CTO", "GeneralCounsel", "ChairOfBoard",
    "BoardMember", "Other", "none",
]
InsiderStatus = Literal["officer", "director", "ten_percent_holder", "none"]

# (entity type, attribute key) -> allowed values. Keys are optional: an absent
# key is "unknown" per the prompt; a present key must be in the vocabulary.
_ATTRIBUTE_VOCAB: dict[tuple[str, str], tuple[str, ...]] = {
    ("LegalProceeding", "status"): get_args(LegalStatus),
    ("GuidanceUpdate", "update_type"): get_args(UpdateType),
    ("CapitalReturn", "return_type"): get_args(ReturnType),
    ("CapitalStructureEvent", "instrument_type"): get_args(InstrumentType),
    ("ESGCommitment", "commitment_type"): get_args(CommitmentType),
    ("MaterialEvent", "event_type"): get_args(EventType),
    ("Person", "officer_role"): get_args(OfficerRole),
    ("Person", "insider_status"): get_args(InsiderStatus),
}


class _Strict(BaseModel):
    """Base: unknown keys are errors, so schema drift is loud."""

    model_config = ConfigDict(extra="forbid")


class EntityExtract(_Strict):
    """One extracted entity."""

    type: EntityType
    name: str
    aliases: list[str] = Field(default_factory=list)
    identifiers: dict[str, str] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_attribute_vocab(self) -> "EntityExtract":
        for (etype, key), allowed in _ATTRIBUTE_VOCAB.items():
            if self.type == etype and key in self.attributes:
                value = self.attributes[key]
                if value not in allowed:
                    raise ValueError(
                        f"{etype}.attributes.{key}={value!r} not in {list(allowed)}"
                    )
        return self


class RelationshipExtract(_Strict):
    """One typed edge between two named entities."""

    source_name: str
    source_type_entity: EntityType
    target_name: str
    target_type_entity: EntityType
    type: RelationshipType
    attributes: dict[str, Any] = Field(default_factory=dict)
    valid_from: str | None = None
    valid_until: str | None = None


class ChunkExtract(_Strict):
    """A verbatim quotable excerpt attached to an entity."""

    entity_name: str | None = None
    entity_type_entity: EntityType | None = None
    content: str
    chunk_type: ChunkType
    valid_from: str | None = None
    valid_until: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class ExtractionPayload(_Strict):
    """The full model output for one section (or one part of a section)."""

    source_file: str
    source_type: Literal["sec_filing"]
    entities: list[EntityExtract]
    relationships: list[RelationshipExtract]
    chunks: list[ChunkExtract]
