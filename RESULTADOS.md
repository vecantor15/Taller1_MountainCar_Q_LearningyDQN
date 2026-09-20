# Comparación de Q-Learning y DQN en MountainCar-v0

## 1. Protocolo experimental

La comparación se realizó sobre el mismo entorno `MountainCar-v0`. Se utilizó la semilla `42` para la ejecución reproducible. Q-Learning se entrenó durante **20.000 episodios** y DQN durante **2.500 episodios**. La evaluación final se realizó con **100 episodios**, política determinista y semillas diferentes de las utilizadas durante el entrenamiento.

En `MountainCar-v0` la recompensa es `-1` por cada paso. Por esta razón, una recompensa menos negativa representa una trayectoria más corta y, por tanto, un mejor desempeño. El éxito se define como alcanzar la bandera antes del límite de pasos del episodio.

Los tiempos registrados corresponden al equipo donde se realizó la ejecución original y pueden variar entre máquinas.

## 2. Hiperparámetros

| Parámetro | Q-Learning | DQN |
|---|---:|---:|
| Tasa de aprendizaje | 0.1 | 0.001 |
| Gamma | 0.99 | 0.99 |
| Epsilon inicial | 1.0 | 1.0 |
| Epsilon mínimo | 0.01 | 0.01 |
| Decaimiento de epsilon | 0.9995 | 0.995 |
| Discretización | 20 × 20 celdas | No aplica |
| Capas ocultas | No aplica | 2 × 128 unidades |
| Tamaño del lote | No aplica | 64 |
| Memoria de repetición | No aplica | 100.000 transiciones |
| Sincronización de red objetivo | No aplica | Cada 10 episodios |
| Persistencia de acción exploratoria | No aplica | 20 pasos |

## 3. Resultados medidos

| Métrica | Q-Learning | DQN |
|---|---:|---:|
| Episodios de entrenamiento | 20.000 | 2.500 |
| Tiempo de entrenamiento | 37.51 s | 317.85 s |
| Mejor media móvil de 100 episodios | -125.39 | -121.32 |
| Episodio de mejor media | 15.860 | 1.506 |
| Recompensa media en evaluación | -149.23 | **-108.32** |
| Desviación estándar | 16.84 | **9.92** |
| Mejor episodio evaluado | -118 | **-83** |
| Peor episodio evaluado | -171 | **-115** |
| Éxitos | 100/100 | 100/100 |

## 4. Evidencia visual

### Q-Learning

![Evidencia de Q-Learning](results/qlearning_evidence.png)

El agente tabular obtuvo una mejor media móvil de `-125.39` en el episodio `15.860`. En evaluación alcanzó una recompensa media de `-149.23`, desviación estándar de `16.84` y 100/100 episodios exitosos. El mejor episodio terminó en 118 pasos.

### DQN

![Evidencia de DQN](results/dqn_evidence.png)

DQN alcanzó una mejor media móvil de `-121.32` en el episodio `1.506`. En evaluación logró una recompensa media de `-108.32`, desviación estándar de `9.92` y 100/100 episodios exitosos. El mejor episodio terminó en 83 pasos.

### Comparación global

![Curvas de entrenamiento](results/training_curves.svg)

## 5. Análisis comparativo

### Desempeño final

La recompensa media pasó de `-149.23` con Q-Learning a `-108.32` con DQN. Dado que el retorno es el negativo del número de pasos, DQN utilizó aproximadamente **40.91 pasos menos por episodio**, equivalente a una reducción de aproximadamente **27.4 %** frente al promedio de Q-Learning.

La desviación estándar también disminuyó de `16.84` a `9.92`, una reducción aproximada de **41.1 %**. Esto indica que, en esta evaluación, el desempeño del agente DQN fue no solamente mejor en promedio sino también más homogéneo entre episodios.

### Velocidad de aprendizaje y costo computacional

Q-Learning requirió 20.000 episodios para el experimento registrado, mientras que DQN fue entrenado con 2.500, es decir, **87.5 % menos episodios**. Sin embargo, el tiempo total de entrenamiento de DQN fue aproximadamente **8.47 veces mayor**, porque cada transición puede involucrar muestreo del replay buffer, inferencia de dos redes y backpropagation.

Esto muestra que “aprender en menos episodios” no equivale necesariamente a “entrenar más rápido en tiempo de cómputo”. El método tabular tiene actualizaciones extremadamente baratas; DQN realiza actualizaciones más costosas pero aprende una representación continua del valor.

### Estabilidad

Q-Learning presentó oscilaciones asociadas a la exploración y a la discretización. Aun así, resolvió los 100 episodios de evaluación. DQN también presentó variabilidad durante el entrenamiento, por lo que la implementación conserva los pesos correspondientes a la mejor media móvil de 100 episodios observada.

El uso de **experience replay** reduce la correlación entre transiciones consecutivas y permite reutilizar experiencias. La **target network** evita que el objetivo de Bellman cambie exactamente al mismo ritmo que la red que intenta aproximarlo. Estos mecanismos contribuyen a estabilizar el entrenamiento.

### Representación del estado

La principal ventaja de Q-Learning es su simplicidad e interpretabilidad: la tabla permite inspeccionar directamente cada valor aprendido. No obstante, para trabajar con un estado continuo es necesario discretizar posición y velocidad. Dos observaciones distintas pueden terminar representadas por la misma celda, generando pérdida de resolución.

DQN evita esa discretización y procesa directamente las variables continuas. Esto permite generalizar entre estados similares y explica parte de la mejora obtenida en la evaluación. A cambio, introduce una arquitectura neuronal, optimización mediante gradiente, mayor consumo de memoria y sensibilidad a hiperparámetros.

### Exploración en MountainCar

La exploración epsilon-greedy independiente en cada paso resultó poco adecuada para DQN en este entorno. MountainCar necesita secuencias sostenidas de acciones en una dirección para acumular impulso. Por este motivo, la implementación mantiene una acción exploratoria durante 20 pasos. Esta persistencia temporal facilita descubrir trayectorias exitosas durante el entrenamiento.

La evaluación final desactiva la exploración y selecciona siempre la acción con mayor valor estimado.

## 6. Ventajas y limitaciones

| Aspecto | Q-Learning | DQN |
|---|---|---|
| Interpretabilidad | Alta | Media |
| Costo por actualización | Muy bajo | Mayor |
| Estado continuo | Requiere discretización | Se procesa directamente |
| Generalización | Limitada a las celdas visitadas | Generaliza mediante la red |
| Dependencia de hiperparámetros | Moderada | Alta |
| Implementación | Más simple | Más compleja |
| Resultado medio de evaluación | -149.23 | **-108.32** |
| Éxitos | 100/100 | 100/100 |

## 7. Conclusión

Los dos métodos resolvieron el entorno en todos los episodios de evaluación, pero lo hicieron con comportamientos y costos diferentes. Q-Learning constituye una solución adecuada cuando el espacio de estados puede discretizarse sin un crecimiento excesivo de la tabla y cuando se priorizan simplicidad e interpretabilidad.

DQN obtuvo un mejor desempeño final y menor variabilidad, al utilizar directamente las observaciones continuas y aproximar la función de valor-acción mediante una red neuronal. En este experimento redujo en aproximadamente 27.4 % la cantidad media de pasos respecto a Q-Learning. Su principal contrapartida fue el mayor costo computacional y la necesidad de mecanismos adicionales de estabilización.

Por tanto, el experimento evidencia la diferencia fundamental entre el enfoque tabular clásico y Deep Reinforcement Learning: Q-Learning memoriza valores sobre una representación discretizada, mientras que DQN aprende una función parametrizada capaz de generalizar entre estados continuos.

## 8. Archivos de evidencia

- `results/qlearning_training.csv`: recompensa y media móvil por episodio de Q-Learning.
- `results/dqn_training.csv`: recompensa y media móvil por episodio de DQN.
- `results/summary.json`: métricas de entrenamiento y evaluación.
- `results/training_curves.svg`: comparación gráfica de entrenamiento.
- `results/qlearning_evidence.png`: evidencia visual individual de Q-Learning.
- `results/dqn_evidence.png`: evidencia visual individual de DQN.

## 9. Reproducción

```bash
uv sync
uv run python scripts/run_experiments.py
uv run --with matplotlib python scripts/generate_evidence.py
uv run mountaincar load qlearning --eval
uv run mountaincar load dqn --eval
```
