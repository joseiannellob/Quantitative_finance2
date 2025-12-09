import pandas as pd
import numpy as np
import scipy.stats as stats
from .backend import market_prices

def portfolio_volatility(
        df:pd.DataFrame,
        vector_w:np.array
        ) -> float:
   '''
   Calculo de la volatlidad de un portafolio
   de inversiones

    df (pd.DataFrame):
        Dataframe de Retornos del portafolio.
    vector_w (np.array)
        vector de pesos de los instrumentos del portafolio

    Return (float): volatlidad del portafolio
    '''
   
   # matriz varianza covarianza
   m_cov = df.cov()

   # vector transpuesto
   vector_w_t = np.array([vector_w])

   # varianza
   vector_cov = np.dot(m_cov,vector_w_t)
   varianza = np.dot(vector_cov,vector_w)

   # volatilidad
   vol = np.sqrt(varianza)

   return vol

def portfolio_returns(
    tickers: list,
    start: str,
    end: str,) -> pd.DataFrame:
    '''
    descarga desde la base de datos los precios de los
    instrumentos indicando en el rango de fechas

    ticker (list):
    lista de nemos de los instrumentos que componen el portafolio

    start (str):
    fecha de inicio de precios
    
    end (str):
    fecha de fin de precios
    
    Return (pd.DataFrame):
    Dataframe de retornos diarios
    '''
    
    # descarga precios
    df = market_prices(
        start_date=start,
        end_date=end,
        tickers=tickers 
        )
        
    # pivot retornos
    df_pivot = pd.pivot_table(
        date=df
        index= 'FECHA',
        columns= 'TICKERS',
        values='PRECIO_CIERRE',
        aggfunc='max')
    df_pivot = df_pivot.pct_change().dropna()
    return df_pivot

def VaR (sigma:float, confidence:float) -> float:
    '''
    Calculo del Valor en Riesgo (VaR) al nivel de confianza indicado
    con supuesto de media cero por tanto es volatilidad por est cero
    '''

    #estadistico z al nivel de confianza
    z_score = stats.norm.ppf(confidence)

    #VaR
    var = sigma * z_score
    return var




