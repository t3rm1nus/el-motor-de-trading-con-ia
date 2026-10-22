# El motor de trading con IA — código del libro

Repositorio complementario del libro **El motor de trading con IA**.

Este proyecto didáctico muestra un flujo reproducible de investigación cuantitativa: validación OHLCV, construcción de variables, etiquetado temporal, modelos de clasificación, validación walk-forward, control del tamaño de posición, backtesting con costes, métricas y comparación de estrategias.

> **Aviso:** el proyecto es educativo. No envía órdenes ni se conecta a exchanges. Los datos sintéticos permiten reproducir las pruebas, pero no demuestran rentabilidad ni representan condiciones reales de mercado.

## Requisitos

- Python 3.11 o 3.12.
- Windows, macOS o Linux.

## Instalación en Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si utilizas macOS/Linux, crea y activa el entorno con `python3 -m venv .venv` y `source .venv/bin/activate`.

## Ejecutar el experimento

Desde la raíz del repositorio:

```bash
python motor_trading_ia.py --bars 2500 --seed 7 --output salida
```

Por defecto, el programa usa datos sintéticos reproducibles y no requiere Internet ni claves API. Los resultados se escriben en `salida/`.

## Ejecutar las pruebas automatizadas

```bash
python -m pytest -q
```

Las pruebas cubren controles de datos, alineación temporal, causalidad de variables, retraso de ejecución, costes, límites de posición, validación temporal y ejecución integrada.

## Relación con los capítulos

El código principal integra los conceptos de los capítulos 1–18. Los laboratorios explicados en el texto guían al lector desde la preparación de datos y variables hasta el backtest, la evaluación fuera de muestra y la supervisión. El script es una implementación integrada para experimentar; no pretende sustituir todas las variantes de producción discutidas en el libro.

## Reproducibilidad y limitaciones

- Usa la versión de Python indicada y conserva las versiones de dependencias fijadas en `requirements.txt`.
- Los datos sintéticos son adecuados para comprobar que el flujo se ejecuta, no para extraer conclusiones de inversión.
- El ejemplo no incluye todas las particularidades de datos reales: dividendos, splits, calendarios, spread variable, liquidez, financiación, límites de mercado o latencia.
- El bootstrap ilustrativo no conserva toda la dependencia temporal y puede infravalorar riesgos extremos.
- Antes de usar datos reales, valida el sistema por instrumento y periodo, incorpora costes realistas y realiza pruebas independientes.

## Versiones

Las versiones etiquetadas se utilizarán para asociar el código a ediciones concretas del libro. Para la edición impresa, consulta la etiqueta de versión indicada en el propio libro.
