from __future__ import annotations

import numpy as np
import pandas as pd
from gymnasium import spaces

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

class GraphStockTradingEnv(StockTradingEnv):
    """
    StockTradingEnv extension that adds a date-specific graph ID
    to each observation.

    The trading, reward and portfolio accounting logic remains inherited
    from FinRL. Only the observation format is changed.
    """
    def __init__(
        self, 
        *args,
        date_to_graph_id: dict[pd.Timestamp, int],
        graph_dates: list[pd.Timestamp],
        **kwargs,
    )->None:
        
        self.date_to_graph_id={
            pd.Timestamp(date): int(graph_id)
            for date, graph_id in date_to_graph_id.items()
        }
        
        self.graph_dates=[
            pd.Timestamp(date)
            for date in graph_dates
        ]

        super().__init__(*args,**kwargs)

        original_state_dim= int(self.observation_space.shape[0])

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(original_state_dim,),
                    dtype=np.float32,
                ),
                "graph_id": spaces.Box(
                    low=0,
                    high=len(self.graph_dates)-1,
                    shape=(1,),
                    dtype=np.int64,
                )
            }
        )
    
    def _get_current_date(self)-> pd.Timestamp:
        """
        Return the date corresponding to the current FinRL state.
        """
        if "date" in self.data.columns:
            current_date =self.data["date"].iloc[0]
        else:
            current_date =self.data.index[0]
        
        return pd.Timestamp(current_date)
    
    def _get_graph_id(self)-> int:
        current_date= self._get_current_date()

        try:
            return self.date_to_graph_id[current_date]
    
        except KeyError as exc:
            raise KeyError(
                "No Granger graph is available for environment date "
                f"{current_date}."
            ) from exc

    def _wrap_observation(self, state)-> dict[str, np.ndarray]:
        return {
            "state": np.asarray(
                state,
                dtype=np.float32,
            ),
            "graph_id":np.asarray(
                [self._get_graph_id()],
                dtype=np.int64,
            ),
        }
    def reset(self, *, seed=None, options=None):
        reset_result= super().reset(
            seed=seed,
            options=options,
        )

        if isinstance(reset_result, tuple):
            state, info= reset_result
            return self._wrap_observation(state), info
        
        return self._wrap_observation(reset_result)
    
    def step(self, action):
        step_result= super().step(action)

        if len(step_result) ==5:
            state, reward, terminated, truncated, info= step_result

            return(
                self._wrap_observation(state),
                reward,
                terminated,
                truncated,
                info,
            )
        
        state, reward, done, info =step_result

        return(
            self._wrap_observation(state),
            reward,
            done,
            info,
        )
