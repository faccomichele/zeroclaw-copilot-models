#!/usr/bin/env python3
"""
Parse the GitHub Copilot supported models page and display a tabulated report
of model names and their rate (multiplier) filtered by subscription type.

Usage:
    python3 parse_models.py [--subscription PLAN]

Subscription options:
    free, student, pro (default), pro+, business, enterprise
"""

import argparse
import json
import re
import sys
import urllib.request

from bs4 import BeautifulSoup
from tabulate import tabulate

SUPPORTED_MODELS_URL = (
    "https://docs.github.com/en/copilot/reference/ai-models/supported-models"
)

SUBSCRIPTION_CHOICES = ["free", "student", "pro", "pro+", "business", "enterprise"]

# Mapping from subscription type to the column header in the per-plan table
SUBSCRIPTION_TO_PLAN_HEADER = {
    "free": "Copilot Free",
    "student": "Copilot Student",
    "pro": "Copilot Pro",
    "pro+": "Copilot Pro+",
    "business": "Copilot Business",
    "enterprise": "Copilot Enterprise",
}

# Subscriptions that use the "Copilot Free" multiplier column
FREE_SUBSCRIPTIONS = {"free", "student"}


def fetch_rendered_html(url: str) -> str:
    """Fetch the GitHub Docs page and return the rendered article HTML."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(req) as response:
        html = response.read().decode("utf-8")

    # The page is a Next.js app; find the __NEXT_DATA__ script tag
    match = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    if not match:
        raise RuntimeError(
            "Could not find __NEXT_DATA__ in the page. "
            "The page structure may have changed."
        )

    next_data = json.loads(match.group(1))
    rendered = (
        next_data.get("props", {})
        .get("pageProps", {})
        .get("articleContext", {})
        .get("renderedPage", "")
    )
    if not rendered:
        raise RuntimeError(
            "Could not extract rendered article content from __NEXT_DATA__."
        )
    return rendered


def _find_table_by_header(tables, required_headers: list[str]):
    """
    Return the first table whose <thead> contains ALL of the required column headers.

    Header text is normalized by collapsing whitespace to handle cases where
    BeautifulSoup concatenates text across inline elements (e.g. <strong>).

    Raises RuntimeError if no matching table is found.
    """
    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    normalized_required = [normalize(h) for h in required_headers]

    for table in tables:
        thead = table.find("thead")
        if not thead:
            continue
        headers = [
            normalize(th.get_text(separator=" ", strip=True))
            for th in thead.find_all("th")
        ]
        if all(h in headers for h in normalized_required):
            return table, headers
    raise RuntimeError(
        f"Could not find a table with headers {required_headers!r}. "
        "The page structure may have changed."
    )


def parse_tables(rendered_html: str):
    """
    Parse the two relevant tables from the rendered article HTML.

    Tables are identified by their column headers so the script remains robust
    to changes in the number or order of tables on the page.

    Returns:
        per_plan_table: dict mapping model_name -> set of available plan headers
        multipliers_table: dict mapping model_name -> {paid: str, free: str}
    """
    soup = BeautifulSoup(rendered_html, "html.parser")
    tables = soup.find_all("table")

    # Identify the per-plan table by its distinctive column headers
    plan_table, plan_headers = _find_table_by_header(
        tables,
        ["Copilot Free", "Copilot Pro", "Copilot Business"],
    )

    # Identify the multipliers table by its distinctive column headers
    multiplier_table, _mult_headers = _find_table_by_header(
        tables,
        ["Multiplier for paid plans", "Multiplier for Copilot Free"],
    )

    # --- Parse per-plan table ---
    per_plan_table: dict[str, set[str]] = {}

    # plan_headers[0] is "Available models in chat"; plan_headers[1:] are plan names

    for row in plan_table.find("tbody").find_all("tr"):
        th = row.find("th")
        if not th:
            continue
        model_name = th.get_text(separator=" ", strip=True)
        available_plans: set[str] = set()
        cells = row.find_all("td")
        for idx, cell in enumerate(cells):
            header = plan_headers[idx + 1]  # offset by 1 for the row header
            svg = cell.find("svg")
            if svg and svg.get("aria-label") == "Included":
                available_plans.add(header)
        per_plan_table[model_name] = available_plans

    # --- Parse multipliers table ---
    multipliers_table: dict[str, dict[str, str]] = {}

    for row in multiplier_table.find("tbody").find_all("tr"):
        th = row.find("th")
        if not th:
            continue
        model_name = th.get_text(separator=" ", strip=True)
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        paid_multiplier = cells[0].get_text(strip=True)
        free_multiplier = cells[1].get_text(strip=True)
        multipliers_table[model_name] = {
            "paid": paid_multiplier,
            "free": free_multiplier,
        }

    return per_plan_table, multipliers_table


def normalize_model_name(name: str) -> str:
    """Convert model name to lowercase with hyphens instead of spaces."""
    return re.sub(r"\s+", "-", name.lower())


def build_report(
    per_plan_table: dict,
    multipliers_table: dict,
    subscription: str,
) -> list[tuple[str, str]]:
    """
    Build the list of (model_name, rate) tuples for the given subscription.

    For free/student subscriptions the 'Copilot Free' multiplier is used;
    for all paid plans the 'paid plans' multiplier is used.
    """
    plan_header = SUBSCRIPTION_TO_PLAN_HEADER[subscription]
    use_free_multiplier = subscription in FREE_SUBSCRIPTIONS

    rows: list[tuple[str, str]] = []

    for model_name, available_plans in per_plan_table.items():
        if plan_header not in available_plans:
            continue

        multiplier_key = "free" if use_free_multiplier else "paid"
        rate = multipliers_table.get(model_name, {}).get(multiplier_key, "N/A")

        normalized_name = normalize_model_name(model_name)
        rows.append((normalized_name, rate))

    # Sort alphabetically by model name
    rows.sort(key=lambda r: r[0])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse GitHub Copilot supported models and display rates "
            "filtered by subscription type."
        )
    )
    parser.add_argument(
        "--subscription",
        choices=SUBSCRIPTION_CHOICES,
        default="pro",
        metavar="PLAN",
        help=(
            "Subscription type to filter by. "
            f"Options: {', '.join(SUBSCRIPTION_CHOICES)}. "
            "Default: pro"
        ),
    )
    args = parser.parse_args()

    try:
        print(f"Fetching models from {SUPPORTED_MODELS_URL} ...", file=sys.stderr)
        rendered_html = fetch_rendered_html(SUPPORTED_MODELS_URL)
        per_plan_table, multipliers_table = parse_tables(rendered_html)
        report_rows = build_report(per_plan_table, multipliers_table, args.subscription)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not report_rows:
        print(
            f"No models found for subscription type '{args.subscription}'.",
            file=sys.stderr,
        )
        sys.exit(0)

    plan_label = SUBSCRIPTION_TO_PLAN_HEADER[args.subscription]
    rate_column_label = (
        "Rate (Copilot Free)"
        if args.subscription in FREE_SUBSCRIPTIONS
        else "Rate (paid plans)"
    )

    print(f"\nGitHub Copilot models for: {plan_label}\n")
    print(
        tabulate(
            report_rows,
            headers=["Model Name", rate_column_label],
            tablefmt="github",
        )
    )
    print()


if __name__ == "__main__":
    main()
