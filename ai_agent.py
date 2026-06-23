import os
import json
import re
import httpx

OPENROUTER_API_BASE = 'https://openrouter.ai/api/v1'
DEFAULT_MODEL = 'meta-llama/llama-3.1-8b-instruct'

SYSTEM_PROMPT = """You are an expert data analyst. You help users analyze Excel/CSV data.
Given the data summary and a user question, you output Python code using pandas to answer the question.

Rules:
- Use the variable `df` for the DataFrame (already loaded)
- Store the result in a variable named `result`
- For plots, use plt.savefig before showing
- Only output valid Python code wrapped in ```python ... ```
- Keep code simple and efficient
- Use pd, plt, sns as already imported
- For text answers, store a string in result

Data Summary: {summary}
Columns: {columns}
First 5 rows preview: {preview}
"""


class AIAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY', '')

    def is_available(self):
        return bool(self.api_key)

    def _call_openrouter(self, messages, model=DEFAULT_MODEL, temperature=0.1, max_tokens=1000):
        resp = httpx.post(
            f'{OPENROUTER_API_BASE}/chat/completions',
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://github.com/excel-ai-analyzer',
                'X-Title': 'Excel AI Analyzer',
            },
            json={'model': model, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']

    def generate_code(self, user_query: str, summary: dict, columns: list, preview: list) -> str:
        if not self.is_available():
            return self._fallback_code_generation(user_query, columns)

        summary_str = json.dumps(summary, indent=2, default=str)
        preview_str = json.dumps(preview, default=str)

        prompt = SYSTEM_PROMPT.format(
            summary=summary_str, columns=columns, preview=preview_str
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query}
        ]

        try:
            content = self._call_openrouter(messages, temperature=0.1, max_tokens=1000)
            code_match = re.search(r'```python\n?(.*?)```', content, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            return content.strip()
        except Exception as e:
            return self._fallback_code_generation(user_query, columns)

    def generate_insights(self, summary: dict, head: list) -> str:
        if not self.is_available():
            return "AI insights require an OpenRouter API key. Set OPENROUTER_API_KEY in .env file."

        messages = [
            {"role": "system", "content": "You are a data analyst. Provide 3-5 key insights about this dataset in bullet points. Be concise and specific."},
            {"role": "user", "content": f"Data summary: {json.dumps(summary, indent=2, default=str)}\nFirst rows: {json.dumps(head[:5], default=str)}"}
        ]

        try:
            return self._call_openrouter(messages, temperature=0.3, max_tokens=500)
        except Exception as e:
            return f"Could not generate insights: {str(e)}"

    def _fallback_code_generation(self, query: str, columns: list) -> str:
        query_lower = query.lower()
        cols_lower = [c.lower() for c in columns]

        if any(w in query_lower for w in ['average', 'mean', 'avg']):
            for i, c in enumerate(cols_lower):
                if any(w in c for w in ['amount', 'price', 'salary', 'value', 'revenue', 'cost', 'spend']):
                    return f"result = df['{columns[i]}'].mean()"
            if len(columns) > 1:
                return f"result = df['{columns[-1]}'].mean()"

        if any(w in query_lower for w in ['total', 'sum']):
            for i, c in enumerate(cols_lower):
                if any(w in c for w in ['amount', 'price', 'salary', 'value', 'revenue', 'cost', 'spend']):
                    return f"result = df['{columns[i]}'].sum()"
            if len(columns) > 1:
                return f"result = df['{columns[-1]}'].sum()"

        if any(w in query_lower for w in ['count', 'how many', 'number of']):
            return f"result = len(df)"

        if any(w in query_lower for w in ['max', 'maximum', 'highest', 'largest']):
            for i, c in enumerate(cols_lower):
                if any(w in c for w in ['amount', 'price', 'salary', 'value', 'revenue', 'cost', 'spend']):
                    return f"result = df['{columns[i]}'].max()"
            if len(columns) > 1:
                return f"result = df['{columns[-1]}'].max()"

        if any(w in query_lower for w in ['min', 'minimum', 'lowest', 'smallest']):
            for i, c in enumerate(cols_lower):
                if any(w in c for w in ['amount', 'price', 'salary', 'value', 'revenue', 'cost', 'spend']):
                    return f"result = df['{columns[i]}'].min()"
            if len(columns) > 1:
                return f"result = df['{columns[-1]}'].min()"

        if any(w in query_lower for w in ['sort', 'order', 'top', 'head', 'first']):
            for i, c in enumerate(cols_lower):
                if any(w in c for w in ['amount', 'price', 'value', 'date', 'time']):
                    return f"result = df.sort_values('{columns[i]}', ascending=False).head(10)"
            return f"result = df.head(10)"

        if any(w in query_lower for w in ['chart', 'plot', 'graph', 'visualize', 'bar', 'pie', 'line']):
            for i, c in enumerate(cols_lower):
                if any(w in c for w in ['category', 'type', 'name', 'description', 'department']):
                    col_idx = i
                    for j, c2 in enumerate(cols_lower):
                        if any(w2 in c2 for w2 in ['amount', 'value', 'count', 'number', 'price']):
                            return (f"plt.figure(figsize=(10,6))\n"
                                    f"df.groupby('{columns[col_idx]}')['{columns[j]}'].sum().plot(kind='bar')\n"
                                    f"plt.title('{columns[j]} by {columns[col_idx]}')\n"
                                    f"plt.xticks(rotation=45)\n"
                                    f"plt.tight_layout()\n"
                                    f"result = 'Chart generated'")
                    return (f"plt.figure(figsize=(10,6))\n"
                            f"df['{columns[col_idx]}'].value_counts().plot(kind='bar')\n"
                            f"plt.title('Distribution of {columns[col_idx]}')\n"
                            f"plt.xticks(rotation=45)\n"
                            f"plt.tight_layout()\n"
                            f"result = 'Chart generated'")

        return f"result = df.head()\nprint('Available columns:', {columns})"
