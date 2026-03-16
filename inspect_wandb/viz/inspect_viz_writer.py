from inspect_viz import Component
from inspect_viz.plot import write_png_async
import logging
from inspect_viz.view import scores_heatmap
from inspect_viz import Data
from pathlib import Path
import pandas as pd
from wandb import Run, log, Image
from inspect_ai.hooks import RunEnd
from inspect_ai.analysis import evals_df

logger = logging.getLogger(__name__)

class InspectVizWriter:

    async def log_scores_heatmap(self, data: RunEnd, run: Run) -> None:
        try:
            logs = [log.location for log in data.logs]
            run.config["logs"] = logs
            df = evals_df(logs)
            await self._log_scores_heatmap(data, df)
        except Exception as e:
            logger.warning(f"Error creating scores heatmap: {e}")

    async def _log_scores_heatmap(self, data: RunEnd, df: pd.DataFrame) -> None:
        viz_data = Data.from_dataframe(df)
        plot = scores_heatmap(viz_data, task_name="task_display_name", model_name="model", score_value="score_headline_value")

        await self._log_image(data.run_id, plot, "scores_heatmap")

    async def _log_image(self, run_id: str, plot: Component, name: str) -> None:
        path = Path("./.plots") / run_id / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        await write_png_async(str(path), plot)
        log({name: Image(str(path))})
