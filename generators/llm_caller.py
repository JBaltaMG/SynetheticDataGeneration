import pandas as pd
import numpy as np
import prompts
import utils.prompt_utils as prompt_utils
import utils.utils as utils
import generators.random_generators as random_generators

def generate_from_prompt(
    company_name: str,
    count: int,
    prompt_type: str,
    *,
    contx: str | None = None,
    model: str = "gpt-4.1",
    temp: float = 0.8,
) -> pd.DataFrame:
    """
    Generic LLM generator using prompts.py templates.
    contx: optional long text (year-end report or other context)
    """

    client = prompt_utils.get_openai_client()
    over_request_count = int(np.floor(int(count) * 1.4))

    if prompt_type not in prompts.PROMPT_LIBRARY:
        raise ValueError(f"Unknown prompt_type '{prompt_type}'")

    # Build dynamic parts
    header = "name;proportionality;unit_price"  # default header
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)

    # If no context provided, keep it blank
    contx_str = f"\nFor context:\n{contx}\n" if contx else ""

    # Fill template
    prompt_template = prompts.PROMPT_LIBRARY[prompt_type]
    full_prompt = prompt_template.format(
        company_name=company_name,
        count=over_request_count,
        contx=contx_str,
        constraints=constraints,
    )

    # LLM call
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful data analyst and domain expert."},
            {"role": "user", "content": full_prompt},
        ],
        temperature=temp,
    )

    # Parse CSV
    df = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)

    return df