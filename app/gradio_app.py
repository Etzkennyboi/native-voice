"""NaturalVoice AI — Phase 1 baseline TTS app (Gradio).

PRD §7 Phase 1 deliverable: script input → generate → progress/errors →
playback → WAV download, with text validation and per-run metadata logging.

Run:  python app/gradio_app.py
(Colab: open it as a notebook cell or just run this file — pass share=True
in main() below if you need the public proxy link.)

SECURITY / PRIVACY (PRD §10):
    - No credentials are read, embedded, or printed anywhere in this app.
    - Users never see stack traces or internal filesystem paths; full
      diagnostics go to the application log only.
    - Generated audio is stored locally under `outputs/` with its script
      settings; delete the folder (or the `--delete-history` flag output)
      to remove all persisted user content.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow `python app/gradio_app.py` from any working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gradio as gr  # noqa: E402

from app.tts_engine import get_engine  # noqa: E402
from app.validation import ValidationError, sanitize_seed, validate_script  # noqa: E402

LOG_FILE = REPO_ROOT / "logs" / "gradio_app.log"


def configure_logging() -> None:
    """Log diagnostics to console + file; never to the end user (PRD §10)."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, encoding="utf-8")],
        force=True,
    )


def _history_table(engine) -> list[list]:
    rows = []
    for rec in engine.recent_runs(limit=12):
        rows.append(
            [
                rec.get("timestamp", ""),
                rec.get("status", ""),
                rec.get("input_chars", ""),
                rec.get("output_seconds", ""),
                rec.get("gen_seconds", ""),
                rec.get("seed", rec.get("settings", {}).get("seed", "")),
                rec.get("audio_file", rec.get("error", "")),
            ]
        )
    return rows


def generate(script: str, cfg_value: float, timesteps: int, seed: str, narration_style: str, progress=gr.Progress()):
    """Gradio handler. Validates input, generates audio, updates all outputs.

    Raises gr.Error with a user-safe message on any failure — Gradio shows
    that message in the UI and we never leak stack traces (PRD §7/§10).
    """
    engine = get_engine()
    progress(0.05, desc="Validating script…")

    try:
        script_clean = validate_script(script, max_chars=int(engine.cfg.get("limits", {}).get("max_script_chars", 5000)))
        # Dropdown value "none (raw model voice)" means: no wrapping tag applied.
        style = "" if narration_style.startswith("none") else narration_style
        settings_overrides = {
            "cfg_value": float(cfg_value),
            "inference_timesteps": int(timesteps),
            "seed": sanitize_seed(seed),
            "narration_style": style,
        }
    except ValidationError as exc:
        raise gr.Error(str(exc)) from None

    try:
        record = engine.generate(script_clean, overrides=settings_overrides)
    except ValidationError as exc:
        engine.log_failure(script, str(exc))
        raise gr.Error(str(exc)) from None  # already user-safe
    except Exception:
        logging.getLogger("naturalvoice.user_request").exception("Generation failed.")
        engine.log_failure(script, "model generation failed (see server logs)")
        raise gr.Error(
            "Generation failed. The problem has been logged on the server — "
            "please try again, possibly with a shorter script."
        ) from None

    progress(1.0, desc="Done")
    audio_path = str(REPO_ROOT / engine.cfg["paths"]["outputs_dir"] / record["audio_file"])
    meta = (
        f"**Model:** `{record['model_id']}` · revision `{record['checkpoint_revision']}`  \n"
        f"**Run:** `{record['run_id']}` @ `{record['timestamp']}` · device `{record['device']}`  \n"
        f"**Settings:** {record['settings']}  \n"
        f"**Timing:** generation {record['gen_seconds']}s → clips {record['output_seconds']}s "
        f"(RTF {record['realtime_factor']})"
    )
    return audio_path, audio_path, meta, _history_table(engine)


def _gr_audio(label: str) -> gr.Audio:
    """gradio 5.x exposed show_download_button; 6.x removed it (download is built in)."""
    try:
        return gr.Audio(label=label, type="filepath", show_download_button=True)
    except TypeError:
        return gr.Audio(label=label, type="filepath")


def build_ui() -> gr.Blocks:
    limits = get_engine().cfg.get("limits", {})
    max_chars = int(limits.get("max_script_chars", 5000))
    infer_cfg = get_engine().cfg.get("inference", {})

    with gr.Blocks(title="NaturalVoice AI — Baseline TTS") as demo:
        gr.Markdown(
            "## NaturalVoice AI — Baseline TTS (Phase 1 MVP)\n"
            "Paste a narration script, press **Generate narration**, then play or "
            "download the WAV. Long scripts are split at sentence boundaries automatically."
        )
        with gr.Row():
            with gr.Column(scale=3):
                script_in = gr.Textbox(
                    label="Narration script",
                    placeholder="The ocean covers more than seventy percent of Earth's surface…",
                    lines=8,
                    max_lines=24,
                )
                generate_btn = gr.Button("Generate narration", variant="primary")
                # Narration style — VoxCPM2's NATIVE control tokens, verified
                # human-sounding in the 2026-10-16 probe (experiments/2026-10-16_phase0-1.md).
                # These wrap text as "(style){text}" before the model sees it, so a plain
                # script gains a human narrator delivery by default.
                style_dd = gr.Dropdown(
                    choices=["documentary", "narrator", "calm", "warm", "slow", "none (raw model voice)"],
                    value="documentary",
                    label="Narration style",
                    info="Wrap the script in a verified narration control token "
                         "(native model feature, not simulated).",
                )
                with gr.Accordion("Advanced generation settings", open=False):
                    cfg_slider = gr.Slider(
                        1.0, 5.0, value=float(infer_cfg.get("cfg_value", 2.0)), step=0.1,
                        label="CFG guidance (cfg_value)",
                        info="Higher = more compliant to the text, can sound stiffer.",
                    )
                    steps_slider = gr.Slider(
                        4, 32, value=int(infer_cfg.get("inference_timesteps", 10)), step=1,
                        label="Inference timesteps",
                        info="Higher = slower, potentially a bit more faithful.",
                    )
                    seed_in = gr.Textbox(
                        label="Seed (optional, blank = random)",
                        placeholder="e.g. 3407",
                    )
            with gr.Column(scale=2):
                audio_out = _gr_audio("Generated audio")
                download_out = gr.DownloadButton(label="Download WAV", visible=False)
                meta_out = gr.Markdown("Generation metadata will appear here.")

        history = gr.Dataframe(
            headers=["timestamp", "status", "chars", "out (s)", "gen (s)", "seed", "file"],
            label="Recent generations (local run log — outputs/run_log.jsonl)",
            value=[],
            interactive=False,
        )

        generate_btn.click(
            fn=generate,
            inputs=[script_in, cfg_slider, steps_slider, seed_in, style_dd],
            outputs=[audio_out, download_out, meta_out, history],
        )

        # Show the download button only once a generation succeeded.
        audio_out.change(lambda: gr.DownloadButton(visible=True), None, download_out)

        gr.Markdown(
            "ℹ️ **Where is my data?** Scripts and generated audio stay on this machine "
            "(`outputs/` + `outputs/run_log.jsonl`). Nothing is uploaded to a third party by "
            "this app, and you can delete those files at any time."
        )
    return demo


def main() -> None:
    configure_logging()
    demo = build_ui()
    # Colab: add `share=True` for a public link; locally keep defaults.
    demo.launch()


if __name__ == "__main__":
    main()
