# Analise peer review

Mostras analizadas: 200.
Predicions identicas entre solo_dudoso e conservador: True.
Como ambos perfis comparten inferencia, o sistema avaliase unha soa vez; os perfis usanse para estudar se os flags de revision capturan maior variabilidade.

## Flags de revision

- solo_dudoso: 3 mostras marcadas.
- conservador: 15 mostras marcadas.
- adicionais de conservador fronte a solo_dudoso: 12.

## Comparacions principais

| Comparacion | HPI MAE | HPI within1 | HPI kappa quad. | IVR app F1 | IVR cond MAE | IVR cond within1 |
| --- | --- | --- | --- | --- | --- | --- |
| sara vs nerea | 0.190 | 0.995 | 0.968 | 0.965 | 0.500 | 0.912 |
| original vs sara | 0.465 | 0.945 | 0.912 | 0.918 | 1.593 | 0.619 |
| original vs nerea | 0.525 | 0.945 | 0.895 | 0.879 | 1.888 | 0.560 |
| original vs system | 0.360 | 0.955 | 0.903 | 0.882 | 1.029 | 0.714 |
| sara vs system | 0.635 | 0.910 | 0.837 | 0.872 | 2.027 | 0.545 |
| nerea vs system | 0.615 | 0.910 | 0.834 | 0.860 | 2.263 | 0.456 |

## Variabilidade nas zonas marcadas

| Subset | n | HPI pair MAE | HPI any disag. | IVR app disag. | IVR pair MAE |
| --- | --- | --- | --- | --- | --- |
| all | 200 | 0.393 | 0.495 | 0.160 | 1.443 |
| solo_dudoso_flagged | 3 | 0.222 | 0.333 | 0.333 | 1.286 |
| conservador_flagged | 15 | 0.578 | 0.600 | 0.400 | 2.163 |
| conservador_extra | 12 | 0.667 | 0.667 | 0.417 | 2.333 |
| safe_in_conservador | 185 | 0.378 | 0.486 | 0.141 | 1.387 |
| safe_in_solo_dudoso | 197 | 0.396 | 0.497 | 0.157 | 1.445 |

## Consistencia HPI-IVR por avaliador

| Avaliador | n | Accuracy | Incons. | IVR>0 con HPI non aplic. | IVR=0 con HPI aplic. |
| --- | --- | --- | --- | --- | --- |
| original | 200 | 1.000 | 0 | 0 | 0 |
| sara | 198 | 1.000 | 0 | 0 | 0 |
| nerea | 199 | 1.000 | 0 | 0 | 0 |
| system | 200 | 1.000 | 0 | 0 | 0 |

## Bootstrap IC 95%

| Metrica | Media | IC low | IC high |
| --- | --- | --- | --- |
| human_variability_hpi_pairwise_mae | 0.394 | 0.333 | 0.460 |
| human_variability_ivr_app_disagreement_rate | 0.160 | 0.110 | 0.215 |
| system_vs_original_hpi_mae | 0.361 | 0.255 | 0.480 |
| system_vs_original_ivr_cond_mae | 1.026 | 0.838 | 1.227 |
| sara_vs_nerea_hpi_mae | 0.190 | 0.135 | 0.250 |
| sara_vs_nerea_ivr_cond_mae | 0.501 | 0.331 | 0.701 |

## Lectura

A comparacion relevante non e so sistema contra etiqueta orixinal: hai que contrastar esa distancia coa distancia Sara-Nerea e coas distancias entre cada experta e a etiqueta orixinal. Se o sistema cae nun rango parecido, a evidencia apoia que funciona como un avaliador experto adicional.
Para IVR, as metricas principais son aplicabilidade e nota condicional. A metrica legacy sobre todas as mostras queda gardada nos CSV/JSON, pero mestura a decision de se IVR aplica coa nota 1..7.
