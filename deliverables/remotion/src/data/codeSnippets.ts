/**
 * ─── CODE SNIPPETS FOR PRESENTATION SLIDES ───
 *
 * Contains verified code excerpts from the raiveFlier codebase, used by the
 * CodeExamplesSlide to display real implementation patterns. Extracting these
 * into a data module keeps the slide component focused on layout/animation
 * and makes it easy to update snippets without touching presentation logic.
 *
 * Two key patterns are showcased:
 *   1. interfaceCode — The ILLMProvider abstract base class, demonstrating
 *      the adapter pattern that decouples business logic from vendor SDKs
 *   2. pipelineStateCode — The PipelineState frozen model, demonstrating
 *      immutable state management with Pydantic v2's ConfigDict(frozen=True)
 *
 * Architecture connection: Pure data module consumed by src/slides/CodeExamplesSlide.tsx.
 * Source files: src/interfaces/llm_provider.py and src/models/pipeline.py
 */

/**
 * ILLMProvider interface — the adapter contract.
 * Every LLM provider (OpenAI, Anthropic, Ollama) implements this ABC,
 * allowing the pipeline to swap providers via dependency injection
 * without touching any business logic.
 */
export const interfaceCode = `class ILLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ) -> str: ...

    @abstractmethod
    async def vision_extract(
        self, image_bytes: bytes, prompt: str
    ) -> str: ...

    @abstractmethod
    def supports_vision(self) -> bool: ...

    @abstractmethod
    def get_provider_name(self) -> str: ...`;

/**
 * PipelineState model — immutable state transitions.
 * Uses Pydantic v2 frozen=True so state can only change via
 * model_copy(update={...}), creating a full audit trail of
 * every pipeline phase transition.
 */
export const pipelineStateCode = `class PipelineState(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    flier: FlierImage
    current_phase: PipelinePhase
    ocr_result: OCRResult | None = None
    extracted_entities: ExtractedEntities | None = None
    confirmed_entities: ExtractedEntities | None = None
    research_results: list[ResearchResult] = Field(
        default_factory=list
    )
    interconnection_map: InterconnectionMap | None = None
    progress_percent: float = Field(
        default=0.0, ge=0.0, le=100.0
    )`;
