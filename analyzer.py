import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import re
import os
import math


def clean_nan(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(item) for item in obj]
    return obj


class ExcelAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.file_name = os.path.basename(file_path)
        self.columns = []
        self.shape = (0, 0)
        self.dtypes = {}
        self.summary = {}
        self._load()

    def _load(self):
        if self.file_path.endswith('.csv'):
            self.df = pd.read_csv(self.file_path)
        else:
            self.df = pd.read_excel(self.file_path, sheet_name=None)
            self.sheet_names = list(self.df.keys())
            self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_names[0])
        self.columns = list(self.df.columns)
        self.shape = self.df.shape
        self.dtypes = {str(k): str(v) for k, v in self.df.dtypes.items()}
        self._compute_summary()

    def _compute_summary(self):
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        text_cols = self.df.select_dtypes(include=['object']).columns

        self.summary = clean_nan({
            'shape': {'rows': int(self.shape[0]), 'columns': int(self.shape[1])},
            'columns': self.columns,
            'dtypes': self.dtypes,
            'null_counts': {str(k): int(v) for k, v in self.df.isnull().sum().items()},
            'numeric_summary': {}
        })

        for col in numeric_cols:
            self.summary['numeric_summary'][str(col)] = {
                'min': float(self.df[col].min()) if pd.notna(self.df[col].min()) else None,
                'max': float(self.df[col].max()) if pd.notna(self.df[col].max()) else None,
                'mean': float(self.df[col].mean()) if pd.notna(self.df[col].mean()) else None,
                'median': float(self.df[col].median()) if pd.notna(self.df[col].median()) else None,
                'std': float(self.df[col].std()) if pd.notna(self.df[col].std()) else None
            }

        for col in text_cols:
            if len(self.df) > 0:
                vals = self.df[col].value_counts().head(10).to_dict()
                self.summary[f'{col}_top_values'] = {str(k): int(v) for k, v in vals.items()}

    def get_head(self, n=10):
        records = self.df.head(n).to_dict(orient='records')
        return clean_nan(records)

    def get_preview(self):
        return {
            'summary': self.summary,
            'head': self.get_head(10),
            'sheet_names': getattr(self, 'sheet_names', [self.file_name])
        }

    def execute_code(self, code: str):
        local_vars = {'df': self.df, 'pd': pd, 'plt': plt, 'sns': sns}
        result = None
        is_plot = False
        plot_data = None

        try:
            sanitized = self._sanitize_code(code)
            exec(sanitized, globals(), local_vars)
            result = local_vars.get('result', None)

            if result is None:
                for key in ['result', 'output', 'answer']:
                    if key in local_vars:
                        result = local_vars[key]
                        break

            if 'fig' in local_vars or 'plt' in local_vars:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                plot_data = base64.b64encode(buf.read()).decode('utf-8')
                plt.close('all')
                is_plot = True

            if isinstance(result, pd.DataFrame):
                result = clean_nan(result.to_dict(orient='records'))
            elif isinstance(result, pd.Series):
                result = clean_nan(result.to_dict())
            elif isinstance(result, (int, float)):
                result = None if (isinstance(result, float) and (math.isnan(result) or math.isinf(result))) else result

            return {
                'success': True,
                'result': result,
                'is_plot': is_plot,
                'plot_data': plot_data,
                'code': sanitized
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'code': code}

    def _sanitize_code(self, code: str):
        code = re.sub(r'^```python\s*', '', code)
        code = re.sub(r'^```\s*', '', code)
        code = re.sub(r'\s*```$', '', code)
        code = code.strip()
        return code

    def generate_chart(self, chart_type: str, x_col: str, y_col: str = None, title: str = None):
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            if chart_type == 'bar':
                if y_col:
                    self.df.groupby(x_col)[y_col].sum().plot(kind='bar', ax=ax)
                else:
                    self.df[x_col].value_counts().plot(kind='bar', ax=ax)
            elif chart_type == 'line':
                self.df.plot(x=x_col, y=y_col, kind='line', ax=ax)
            elif chart_type == 'scatter':
                self.df.plot(x=x_col, y=y_col, kind='scatter', ax=ax)
            elif chart_type == 'pie':
                self.df[x_col].value_counts().plot(kind='pie', ax=ax)
            elif chart_type == 'hist':
                self.df[x_col].plot(kind='hist', ax=ax)
            elif chart_type == 'heatmap':
                sns.heatmap(self.df.select_dtypes(include=['number']).corr(), annot=True, ax=ax)
            else:
                return None

            if title:
                ax.set_title(title)

            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode('utf-8')
            plt.close('all')
            return {'plot_data': plot_data, 'chart_type': chart_type}
        except Exception as e:
            plt.close('all')
            return None
