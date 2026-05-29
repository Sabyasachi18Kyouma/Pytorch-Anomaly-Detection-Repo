import ollama

from llm.report_generator import load_log_rows, compute_summary, save_report
from llm.prompt_templates import build_incident_report_prompt


def generate_local_llm_report(summary, model_name="llama3.2:1b"):
    """
    Generate incident report using a local Ollama model.
    No API key required.
    """

    prompt = build_incident_report_prompt(summary)

    response = ollama.generate(
        model=model_name,
        prompt=prompt,
    )

    return response["response"]


def main():
    log_path = "logs/realtime_log_cnn1d.csv"
    output_path = "reports/local_incident_report.md"

    rows = load_log_rows(log_path)
    summary = compute_summary(rows)

    print("Computed log summary:")
    print(summary)

    print("\nGenerating local LLM report using Ollama...")

    report_text = generate_local_llm_report(
        summary=summary,
        model_name="llama3.2:1b",
    )

    save_report(report_text, output_path)


if __name__ == "__main__":
    main()