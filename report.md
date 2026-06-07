# Wissenschaftlicher Projektbericht: Erkennung von Gebärdensprachen-Buchstaben mit Machine Learning

## Projektdaten

| Feld | Angabe |
|---|---|
| Modul | Machine Learning |
| Semester | SoSe 2026 |
| Dozent | Nicola Fanton |
| Projekttitel | Sign Language Recognition / ASL-Alphabet-Erkennung |
| Abgabedatum | 07.06.2026 |

---

## Gruppenmitglieder

| Name | Matrikelnummer | Rolle im Projekt |
|---|---:|---|
| Ervin Omerovic | 209641 | Technische Umsetzung der Datenpipeline, Projektstruktur, Integration der Datensätze, Erweiterung der Datenbasis durch eigene Webcam-Aufnahmen, Modelltraining, Evaluation, Webcam-Anwendung, README und Bericht |
| Delil Yilmaz | 204114 | Recherche und Auswahl geeigneter Datensätze, Unterstützung bei Datenstruktur und Datenaufbereitung, Kontrolle einzelner Skripte zur Datenaufbereitung, Prüfung von Trainingskurven und Confusion Matrix, Bewertung der Fehlklassifikationen, Berichtskontrolle |

## 1. Einleitung und Zieldefinition

Die automatische Erkennung von Gebärdensprachen-Buchstaben ist ein relevantes Anwendungsfeld des maschinellen Lernens, da sie die digitale Unterstützung der Kommunikation mit Gebärdensprache ermöglichen kann. In diesem Projekt wird untersucht, wie ausgewählte statische Buchstaben der American Sign Language (ASL) mithilfe eines bildbasierten Machine-Learning-Modells erkannt werden können. Der Fokus liegt dabei auf acht ausgewählten Klassen: **A, B, C, L, V, W, O und Y**.

Ziel des Projekts ist die Entwicklung eines CNN-basierten Klassifikationsmodells, das Gebärdensprachen-Buchstaben anhand von Bilddaten erkennt und zusätzlich in einer einfachen Webcam-Anwendung praktisch demonstriert werden kann. Neben der reinen Modellgenauigkeit wird besonderer Wert auf eine nachvollziehbare Datenpipeline, eine saubere Datenbereinigung, eine externe Evaluation und eine reproduzierbare Projektdokumentation gelegt.

**Forschungsfrage:**  
Wie zuverlässig kann ein CNN-basiertes Modell ausgewählte Gebärdensprachen-Buchstaben anhand von Bild- und Webcam-Daten sowie geeigneter Evaluationsmetriken klassifizieren?

Zur Beantwortung der Forschungsfrage werden unter anderem Accuracy, Precision, Recall, F1-Score und eine Confusion Matrix herangezogen. Dadurch wird nicht nur die Gesamtleistung des Modells bewertet, sondern auch analysiert, welche Klassen besonders zuverlässig erkannt werden und bei welchen Gebärden häufiger Fehlklassifikationen auftreten.

## 2. Theoretischer Hintergrund

### 2.1 Bildklassifikation

Bildklassifikation beschreibt die automatische Zuordnung eines Bildes zu einer vordefinierten Klasse (Daroya et al., 2018, S. 1). Im Projekt wird dieser Ansatz genutzt, da jeder Gebärdensprachen-Buchstabe als eigene Klasse betrachtet wird. Das Modell soll anhand eines Bildes oder Webcam-Frames erkennen, ob beispielsweise A, B, C, L, V, W, O oder Y gezeigt wird. Damit handelt es sich um ein Mehrklassen-Klassifikationsproblem.

### 2.2 Convolutional Neural Networks

Convolutional Neural Networks (CNNs) sind neuronale Netze, die besonders für Bilddaten geeignet sind, weil sie lokale Muster wie Kanten, Formen und Texturen erkennen können. Für Gebärdensprachen-Buchstaben ist das relevant, da sich die Klassen häufig durch kleine Unterschiede in Fingerstellung, Handform oder Silhouette unterscheiden. Ein CNN kann solche visuellen Merkmale automatisch aus den Trainingsdaten lernen und ist daher für die vorliegende Klassifikationsaufgabe geeignet (LeCun et al., 1998, S. 5).

### 2.3 Transfer Learning

Transfer Learning bedeutet, dass ein bereits vortrainiertes Modell für eine neue Aufgabe weiterverwendet wird. Dadurch muss das Modell nicht alle visuellen Merkmale von Grund auf neu lernen. In diesem Projekt wird MobileNetV2 verwendet, da diese Architektur eine gute Balance zwischen Genauigkeit und Rechenaufwand bietet und sich deshalb für eine spätere Echtzeit-Anwendung über die Webcam eignet (Tan et al., 2018, S. 2).

### 2.4 Data Cleaning und Qualitätssicherung

Die Qualität der Trainingsdaten hat einen großen Einfluss auf die Modellleistung. Fehlerhafte, doppelte oder unscharfe Bilder können dazu führen, dass das Modell falsche Muster lernt oder zu stark auf bestimmte Datenartefakte reagiert (Côté et al., 2024, S. 1; 14). Deshalb werden die Daten vor dem Training mit `clean_dataset.py` bereinigt.

Die Datenbereinigung umfasst unter anderem:

- Entfernung von Duplikaten mithilfe von Perceptual Hashing
- Filterung unscharfer Bilder über Laplacian-Varianz
- Entfernung beschädigter oder nicht lesbarer Dateien
- Filterung nahezu einfarbiger oder extremer Bildinhalte
- Erhalt visueller Variation, z. B. unterschiedliche Lichtverhältnisse und Perspektiven

Diese Schritte sollen die Datenbasis verlässlicher machen und die Grundlage für ein stabileres Training schaffen.

### 2.5 Data Augmentation

Data Augmentation beschreibt die künstliche Veränderung vorhandener Trainingsbilder, zum Beispiel durch Rotation, Helligkeitsänderung, Zoom oder Spiegelung. Dadurch sieht das Modell während des Trainings mehr Variationen derselben Klasse. Für dieses Projekt ist das wichtig, weil Webcam-Bilder je nach Licht, Position und Kameraeinstellung unterschiedlich aussehen können. Die Augmentation soll das Modell robuster gegenüber solchen Veränderungen machen (Shorten & Khoshgoftaar, 2019, S. 4).

### 2.6 Herausforderungen bei der Handgestenerkennung

Die Erkennung von Handgesten ist anspruchsvoll, da unterschiedliche Faktoren die Vorhersage beeinflussen können. Dazu gehören Beleuchtung, Hintergrund, Kamerawinkel, Handgröße, Bildqualität und mögliche Spiegelungen durch die Webcam. Zusätzlich sehen sich einige Gebärden visuell ähnlich, wodurch Fehlklassifikationen entstehen können. Diese Herausforderungen werden im Projekt durch Datenbereinigung, Data Augmentation, externe Testdaten und eine praktische Webcam-Anwendung berücksichtigt.

## 3. Datenbasis und Datenmanagement

### 3.1 Verwendete Datensätze

Das Projekt verwendet drei externe Datensätze:

- Datensatz 1 (Kaggle): [ASL Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)
- Datensatz 2 (Zenodo): [ASL Dataset](https://zenodo.org/records/14635573)
- Datensatz 3 (Kaggle, synthetisch): [Synthetic ASL Alphabet Dataset](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet)

Aus diesen Datensätzen werden ausschließlich die acht Klassen **A, B, C, L, V, W, O und Y** verwendet. Die Auswahl beschränkt den Projektumfang bewusst, damit Training, Evaluation und Webcam-Demo innerhalb des Modulprojekts realistisch umgesetzt werden können.

Zusätzlich zu den externen Datensätzen wurde die Datenbasis durch eigene Webcam-Aufnahmen erweitert. Dafür wurden pro Buchstabe ungefähr **150 bis 200 zusätzliche Bilder** aufgenommen. Diese eigenen Bilder dienen dazu, das Modell stärker an realistische Aufnahmebedingungen der späteren Webcam-Anwendung anzupassen. Während die externen Datensätze eine breite Grundlage für das Training liefern, bilden die eigenen Aufnahmen typische Bedingungen der praktischen Nutzung ab, etwa Kameraqualität, Beleuchtung, Handposition und Hintergrund.

### 3.2 Datenstruktur

Die Projektstruktur trennt Rohdaten, bereinigte Daten und externe Testdaten:

```text
data_raw/        # Rohdaten
data_cleaned/    # bereinigte Trainingsdaten
external_test/   # externe Testdaten
```

Die Trainingsdaten werden aus `data_cleaned/` geladen. Die Rohdaten in `data_raw/` dienen als Ausgangsbasis für die Datenbereinigung. Die externen Testdaten unter `external_test/` werden nicht für das Training verwendet, sondern ausschließlich für die finale Evaluation.

### 3.3 Datenaufbereitung

Die Bereinigung erfolgt mit dem Skript `clean_dataset.py`. Dabei werden unter anderem doppelte, unscharfe oder beschädigte Bilder entfernt. Anschließend werden die Bilddaten für das Training vorbereitet und durch Data Augmentation erweitert. Dazu gehören:

- horizontaler Flip
- Rotation um bis zu ±15°
- Helligkeitsvariation
- Zoom- und Verschiebungsoperationen

Die Aufbereitung soll sicherstellen, dass das Modell nicht nur auf sehr einheitliche Trainingsbilder reagiert, sondern auch mit leichten Variationen umgehen kann.

Die eigenen Webcam-Bilder wurden in die bestehende Klassenstruktur integriert und anschließend ebenfalls durch die Datenbereinigung verarbeitet. Dadurch werden diese Aufnahmen nicht separat behandelt, sondern gemeinsam mit den externen Trainingsdaten in die Pipeline aufgenommen. Der Vorteil besteht darin, dass das Modell nicht nur auf kuratierte Datensatzbilder trainiert wird, sondern auch Beispiele aus der tatsächlichen Einsatzumgebung sieht. Dies ist besonders wichtig, da bei der Live-Erkennung über die Webcam andere Bedingungen auftreten können als in den externen Datensätzen.

Für die Aufnahme der eigenen Bilder wurde ein Webcam-Capture-Ansatz genutzt. Dabei wird die jeweilige Handgeste vor der Kamera gezeigt und der relevante Bildbereich als Trainingsbild gespeichert. Dieses Vorgehen eignet sich besonders, um zusätzliche Beispiele für schwierige oder häufig verwechselte Klassen aufzunehmen und dadurch die Datenbasis gezielt zu erweitern.

## 4. Beschreibung der Methodik

### 4.1 Trainingspipeline

Die ursprüngliche Pipeline wurde so angepasst, dass die Daten zunächst bereinigt und erst danach für Training und Validierung genutzt werden:

```text
externe Datensätze + eigene Webcam-Aufnahmen → data_raw/ → clean_dataset.py → data_cleaned/ → Training und Validierung
                                                                                              ↓
                                                                                      external_test/ → finale Evaluation
```

Diese Struktur wurde gewählt, weil die Datenqualität ein zentraler Faktor für die spätere Modellleistung ist. Durch die vorgelagerte Bereinigung wird offensichtliches Rauschen reduziert, bevor das Modell trainiert wird. Die zusätzlichen eigenen Webcam-Aufnahmen erweitern die Datenbasis um Beispiele aus der späteren Anwendungssituation und sollen dadurch die Robustheit bei der Live-Erkennung verbessern.

### 4.2 Modellwahl

Für die Klassifikation wird ein CNN-basierter Ansatz mit Transfer Learning verwendet. Als Basisarchitektur kommt MobileNetV2 zum Einsatz. Diese Entscheidung wurde getroffen, weil CNNs für Bildklassifikationsaufgaben gut geeignet sind und MobileNetV2 vergleichsweise effizient arbeitet. Dies ist besonders relevant, da das Modell nicht nur offline evaluiert, sondern auch in einer Webcam-Demo eingesetzt wird.

### 4.3 Begründung der getroffenen Entscheidungen

Die wichtigsten Entscheidungen im Projekt lassen sich wie folgt begründen:

| Entscheidung | Begründung |
|---|---|
| Auswahl von acht Klassen | Begrenzung des Projektumfangs und bessere Kontrollierbarkeit der Evaluation |
| Nutzung eines CNN-basierten Modells | Geeignet für visuelle Mustererkennung in Bilddaten |
| Einsatz von Transfer Learning | Effizienteres Training und Nutzung vortrainierter Merkmalsrepräsentationen |
| Datenbereinigung vor dem Training | Reduktion von Duplikaten, Unschärfe und beschädigten Dateien |
| Externe Testdaten | Realistischere Bewertung der Generalisierungsfähigkeit |
| Webcam-Demo | Praktische Demonstration der Anwendbarkeit des Modells |
| Eigene Webcam-Aufnahmen | Ergänzung der externen Datensätze um realistische Beispiele aus der späteren Anwendungssituation |
| Capture-Ansatz für eigene Bilder | Gezielte Aufnahme zusätzlicher Trainingsdaten pro Klasse, insbesondere zur Verbesserung der Praxistauglichkeit |

### 4.4 Evaluation

Zur Bewertung des Modells werden mehrere Metriken verwendet:

- **Accuracy** zur Bewertung der Gesamtklassifikation
- **Precision** zur Analyse falsch positiver Vorhersagen
- **Recall** zur Analyse übersehener Klassen
- **F1-Score** als kombinierte Bewertung aus Precision und Recall
- **Confusion Matrix** zur Visualisierung von Fehlklassifikationen zwischen Klassen

Die finale Evaluation erfolgt auf externen Testdaten, die nicht im Training verwendet wurden. Dadurch soll verhindert werden, dass die Modellleistung nur auf interne Validierungsdaten bezogen wird.

## 5. Ergebnisse und Erkenntnisse

In diesem Kapitel werden die zentralen Ergebnisse des Projekts in der Reihenfolge des tatsächlichen Projektablaufs dargestellt. Zunächst wird die Datenbereinigung beschrieben, da sie die Grundlage für das anschließende Training bildet. Danach folgen Trainingsverlauf, externe Evaluation, per-Klasse-Ergebnisse und Confusion Matrix.

### 5.1 Datenbereinigung und Datenverteilung

Vor dem Training wurde eine systematische Datenbereinigung durchgeführt. Die aktuellste dokumentierte Cleaning-Session wurde am **07.06.2026 um 13:46:49** durchgeführt. Dabei wurden insgesamt **15.095 Bilder** verarbeitet. Davon wurden **13.398 Bilder** in den bereinigten Datensatz übernommen und **1.697 Bilder** durch die Cleaning-Pipeline entfernt. Die Erfolgsquote der Bereinigung lag damit bei **88,8 %**.

| Kennzahl | Wert |
|---|---:|
| Gesamt verarbeitet | 15.095 Bilder |
| Gespeichert | 13.398 Bilder |
| Entfernt | 1.697 Bilder |
| Erfolgsquote | 88,8 % |
| Blur Threshold | 40,0 |
| Mindestgröße | 50 Pixel |
| Deduplizierung | aktiviert |

Die Entfernung erfolgte auf Basis der im Cleaning-Skript definierten Qualitätskriterien. Dazu zählen insbesondere die Duplikaterkennung, die Filterung unscharfer Bilder über die Laplacian-Varianz, die Mindestgröße von 50 Pixeln sowie die Prüfung auf beschädigte oder ungeeignete Bilddateien. Da die Datenbasis zuvor zusätzlich um eigene Webcam-Aufnahmen erweitert wurde, umfasst dieser Cleaning-Lauf sowohl externe Datensatzbilder als auch selbst aufgenommene Bilder. Die verbleibenden **13.398 bereinigten Bilder** bilden die Grundlage für das anschließende Training.

![Datenverteilung nach der Bereinigung](report_assets/data_distribution.png)

*Abbildung 1: Verteilung der bereinigten Bilddaten pro Klasse.*

### 5.2 Trainingsverlauf

Nach der Datenbereinigung wurde das Modell auf den bereinigten Trainingsdaten trainiert. Der Trainingsverlauf zeigt die Entwicklung von Accuracy und Loss über die Trainingsepochen hinweg.

![Trainings- und Validierungsverlauf](report_assets/training_curves.png)

*Abbildung 2: Verlauf von Accuracy und Loss während des Trainings.*

Der Trainingsverlauf zeigt eine schnelle Verbesserung in den ersten Epochen und eine anschließende Stabilisierung auf hohem Niveau. Aus den aktuellen Epoch-Metriken ergeben sich folgende Werte:

- Maximale Validierungsgenauigkeit: **0.9888**
- Validierungsgenauigkeit am Ende der aufgezeichneten Epochen: **0.9877**
- Trainingsgenauigkeit in der letzten Epoche: **0.9931**

Die Kurven zeigen, dass das Modell bereits nach wenigen Epochen eine hohe Genauigkeit erreicht. Der Loss sinkt sowohl im Training als auch in der Validierung deutlich ab. Gegen Ende des Trainings bleibt die Validierungsgenauigkeit stabil, während die Trainingsgenauigkeit nur leicht höher liegt. Dies spricht für ein insgesamt stabiles Training ohne stark ausgeprägtes Overfitting.

### 5.3 Übersicht der Evaluationsmetriken

Nach dem Training wurde das Modell auf externen Testdaten evaluiert. Diese Testdaten wurden nicht für das Training verwendet und ermöglichen daher eine realistischere Einschätzung der Generalisierungsfähigkeit.

Die finale externe Evaluation liefert folgende Gesamtmetriken:

| Metrik | Wert |
|---|---:|
| Accuracy | 0.9179 |
| Precision | 0.9209 |
| Recall | 0.9179 |
| F1-Score | 0.9181 |
| Test-Support | 1.400 Bilder |

Die externe Accuracy von **0.9179** zeigt, dass das Modell rund **91,8 %** der externen Testbilder korrekt klassifiziert. Precision, Recall und F1-Score liegen ebenfalls auf einem ähnlichen Niveau. Dadurch zeigt sich, dass das Modell insgesamt eine solide Leistung erzielt, einzelne Klassen jedoch unterschiedlich zuverlässig erkannt werden.

### 5.4 Ergebnisse pro Klasse

Neben den Gesamtmetriken wurden auch die einzelnen Klassen separat ausgewertet. Dadurch lässt sich erkennen, welche Gebärdensprachen-Buchstaben besonders zuverlässig erkannt werden und bei welchen Klassen häufiger Fehler auftreten.

| Klasse | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| A | 0.95 | 0.85 | 0.90 | 175 |
| B | 0.94 | 0.98 | 0.96 | 175 |
| C | 0.97 | 0.95 | 0.96 | 175 |
| L | 0.86 | 0.89 | 0.88 | 175 |
| V | 0.92 | 0.92 | 0.92 | 175 |
| W | 0.93 | 0.90 | 0.92 | 175 |
| O | 0.96 | 0.90 | 0.93 | 175 |
| Y | 0.82 | 0.95 | 0.88 | 175 |

Die per-Klasse-Ergebnisse zeigen, dass die Klassen **B** und **C** besonders zuverlässig erkannt werden. Auffällig ist die Klasse **Y**, da sie einen hohen Recall von 0.95, aber eine vergleichsweise niedrige Precision von 0.82 aufweist. Das bedeutet, dass echte Y-Bilder häufig korrekt erkannt werden, andere Klassen jedoch ebenfalls häufiger fälschlich als Y vorhergesagt werden. Auch die Klasse **L** weist mit einer Precision von 0.86 einen vergleichsweise niedrigeren Wert auf, was auf Verwechslungen mit visuell ähnlichen Handgesten hinweist.

### 5.5 Confusion Matrix

Zur genaueren Analyse der Fehlklassifikationen wurde eine Confusion Matrix erstellt. Sie zeigt, welche Klassen korrekt erkannt und welche Klassen miteinander verwechselt wurden.

![Confusion Matrix der externen Evaluation](report_assets/confusion_matrix_external_evaluation.png)

*Abbildung 3: Confusion Matrix der externen Evaluation.*

Die Confusion Matrix bestätigt die per-Klasse-Metriken. Die stark ausgeprägte Diagonale zeigt, dass der Großteil der Bilder korrekt klassifiziert wurde. Gleichzeitig sind einige Fehlklassifikationen erkennbar. Besonders häufig werden **A** und **L** als **Y** klassifiziert. Zudem treten Verwechslungen zwischen **V** und **W** auf, was durch die visuelle Ähnlichkeit der Handgesten erklärbar ist. Auch **O** wird vereinzelt als **L** klassifiziert. Diese Fehlermuster zeigen, dass die Gesamtaccuracy allein nicht ausreicht, sondern durch per-Klasse-Metriken und die Confusion Matrix ergänzt werden muss.

## 6. Diskussion und Interpretation

### 6.1 Modellleistung und Generalisierung

Das Modell erreicht auf den internen Validierungsdaten sehr hohe Werte, während die externe Test-Accuracy bei **0.9179** liegt. Diese Differenz zeigt, dass die externen Testdaten anspruchsvoller sind und eine realistischere Einschätzung der Generalisierungsfähigkeit liefern. Die externe Accuracy von rund **91,8 %** ist für die betrachteten acht Klassen ein gutes Ergebnis, zeigt aber auch, dass das Modell nicht fehlerfrei arbeitet.

Die Generalisierungslücke zwischen den sehr hohen Validierungswerten und den externen Testergebnissen ist nachvollziehbar, da die externen Testdaten aus anderen Quellen stammen und sich in Beleuchtung, Bildqualität, Hintergrund oder Perspektive von den Trainingsdaten unterscheiden können. Gerade bei Handgesten können solche Domänenunterschiede die Klassifikation beeinflussen.

### 6.2 Fehlermuster

Die per-Klasse-Metriken und die Confusion Matrix zeigen, dass nicht alle Klassen gleich zuverlässig erkannt werden. Besonders auffällig ist die Klasse **Y** mit einer niedrigeren Precision von 0.82. Das deutet darauf hin, dass andere Gebärden teilweise fälschlich als Y klassifiziert werden. In der Confusion Matrix ist dies insbesondere bei den Fehlklassifikationen **A → Y** und **L → Y** sichtbar.

Außerdem treten Verwechslungen zwischen **V** und **W** auf. Diese beiden Klassen unterscheiden sich vor allem durch die Anzahl und Stellung der geöffneten Finger. Wenn Fingerabstände, Handwinkel oder Bildschärfe ungünstig sind, können diese Unterschiede für das Modell schwerer erkennbar sein. Auch **O** wird teilweise als **L** klassifiziert, was auf visuelle Überschneidungen in der Handform oder auf Unterschiede in der Bildperspektive zurückzuführen sein kann.

Diese Fehlermuster lassen sich vermutlich durch visuelle Ähnlichkeiten zwischen bestimmten Handgesten, Unterschiede in der Bildqualität und Domänenunterschiede zwischen Trainings- und Testdaten erklären.

### 6.3 Einfluss der Datenbereinigung

Die Datenbereinigung hatte eine wichtige Funktion im Projekt, da sie offensichtliches Rauschen aus den Trainingsdaten entfernt hat. Im aktuellsten Cleaning-Lauf wurden 15.095 Bilder geprüft, 13.398 Bilder gespeichert und 1.697 Bilder aussortiert. Die Entfernung ungeeigneter Bilder reduziert die Gefahr, dass das Modell wiederholte, unscharfe oder fehlerhafte Bildmuster zu stark gewichtet. Dadurch wird die Grundlage für ein stabileres und nachvollziehbareres Training geschaffen.

### 6.4 Einfluss von Transfer Learning, Data Augmentation und eigenen Webcam-Daten

Transfer Learning unterstützt das Training, da bereits gelernte visuelle Merkmale wiederverwendet werden. Dadurch kann das Modell schneller konvergieren und auch mit begrenzter Datenmenge sinnvolle Merkmale lernen. Data Augmentation erweitert die Variation der Trainingsbilder und trägt dazu bei, dass das Modell robuster gegenüber leichten Veränderungen in Beleuchtung, Position und Ausrichtung wird.

Die nachträglich ergänzten eigenen Webcam-Aufnahmen erhöhen zusätzlich die Nähe zwischen Trainingsdaten und praktischer Anwendung. Dadurch werden Bedingungen berücksichtigt, die in externen Datensätzen nicht immer ausreichend abgebildet sind, zum Beispiel konkrete Webcam-Qualität, Raumbeleuchtung, Hintergrund und individuelle Handhaltung. Dies ist besonders relevant, weil das Modell nicht nur auf externen Testdaten, sondern auch in einer Live-Anwendung funktionieren soll.

### 6.5 Praxisanwendbarkeit

Die Webcam-Anwendung zeigt, dass das trainierte Modell nicht nur theoretisch evaluiert, sondern auch praktisch eingesetzt werden kann. Über `predict_webcam.py` wird das Modell geladen, ein Kamerabild verarbeitet und die vorhergesagte Klasse direkt angezeigt. Die praktische Nutzung hängt jedoch weiterhin von äußeren Faktoren ab, zum Beispiel Beleuchtung, Kameraperspektive und Position der Hand.

## 7. Schlussfolgerung

Die Forschungsfrage kann grundsätzlich positiv beantwortet werden. Das CNN-basierte Modell klassifiziert die ausgewählten Gebärdensprachen-Buchstaben auf externen Testdaten mit einer Accuracy von **0.9179** und erreicht auch bei Precision, Recall und F1-Score solide Werte. Damit zeigt das Modell eine insgesamt zuverlässige Leistung für die acht betrachteten Klassen.

Gleichzeitig zeigen die Ergebnisse, dass die Zuverlässigkeit je nach Buchstabe variiert. Besonders die per-Klasse-Metriken und die Confusion Matrix machen deutlich, dass einzelne Klassen häufiger verwechselt werden. Dies betrifft vor allem visuell ähnliche Handgesten sowie Klassen, die unter bestimmten Aufnahmebedingungen schwieriger voneinander zu unterscheiden sind. Für eine praktische Webcam-Anwendung ist das Modell daher grundsätzlich geeignet, bleibt jedoch abhängig von kontrollierten Bedingungen wie guter Beleuchtung, ruhigem Hintergrund, passender Kameraperspektive und klarer Handposition.

Insgesamt liefert das Projekt eine reproduzierbare und praxisnahe Umsetzung einer Gebärdensprachen-Buchstabenerkennung mit Machine Learning. Die Kombination aus Datenbereinigung, Transfer Learning, eigener Erweiterung der Datenbasis, externer Evaluation und Webcam-Demo zeigt, dass der gewählte Ansatz für den Projektumfang angemessen ist.

## 8. Zukünftige Arbeiten

Für zukünftige Arbeiten ergeben sich mehrere Erweiterungsmöglichkeiten:

1. **Erweiterung auf das vollständige Alphabet:** Das Modell könnte auf weitere Buchstaben ausgeweitet werden.
2. **Integration von Bewegungsgebärden:** Buchstaben wie J und Z erfordern Bewegungsinformationen und könnten mit Videodaten behandelt werden.
3. **Bessere Handsegmentierung:** Verfahren wie MediaPipe könnten genutzt werden, um die Handregion zuverlässiger vom Hintergrund zu trennen.
4. **Mehr reale Webcam-Daten:** Die Datenbasis wurde bereits durch eigene Webcam-Aufnahmen erweitert. Zukünftig könnten weitere Aufnahmen unter noch stärker variierenden Licht-, Hintergrund- und Kamerabedingungen die Praxisleistung weiter verbessern.
5. **Optimierung für Echtzeit:** Das Modell könnte für schnellere Inferenz oder mobile Nutzung weiter optimiert werden.

## 9. Reproduzierbarkeit und Setup

### 9.1 Installation

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 9.2 Workflow

```bash
# 1. Optional: eigene Webcam-Bilder aufnehmen
# Beispiel: python capture_dataset_images.py

# 2. Daten bereinigen
python clean_dataset.py

# 3. Modell trainieren
python train_simple.py

# 4. Modell evaluieren
python evaluate.py --model models/sign_language_model.h5

# 5. Echtzeit-Vorhersage starten
python predict_webcam.py
```

### 9.3 Abhängigkeiten

- Python 3.10+
- TensorFlow 2.16.1+
- OpenCV 4.9.0+
- scikit-learn
- NumPy
- Matplotlib
- Pillow
- MediaPipe 0.10.11

### 9.4 Wichtige Verzeichnisse

- `data_raw/`: Rohdaten
- `data_cleaned/`: bereinigte Trainingsdaten
- `external_test/`: externe Testdaten
- `models/`: trainiertes Modell
- `report_assets/`: Abbildungen für den Bericht

## 10. Erklärung zur Nutzung von KI

Im Rahmen des Projekts wurden KI-Assistenzsysteme unterstützend eingesetzt. Die Nutzung erfolgte nicht als Ersatz für die eigenständige Bearbeitung, sondern zur Unterstützung bei Verständnisfragen, sprachlicher Überarbeitung, Strukturierung und punktueller Programmierhilfe. Die fachlichen Entscheidungen, die Auswahl der Methodik, die Durchführung der Experimente, die Bewertung der Ergebnisse und die finale Zusammenstellung des Projekts wurden von der Gruppe verantwortet.

Die KI wurde insbesondere für folgende Zwecke genutzt:

- Erklärung und Einordnung einzelner Machine-Learning-Konzepte, z. B. CNNs, Transfer Learning, Data Augmentation und Confusion Matrix
- sprachliche Überarbeitung einzelner Berichtsteile und Verbesserung der wissenschaftlichen Formulierungen
- Unterstützung bei der Strukturierung des Projektberichts und der README-Datei
- Hilfe bei der Fehlersuche, z. B. bei Python-Umgebungen, Paketinstallationen, Git/GitHub und Modellpfaden
- punktuelle Programmierhilfe und Kommentierung bestehender Skripte, ohne die Projektlogik vollständig automatisiert erstellen zu lassen
- Formulierung von Prompts zur gezielten Weiterentwicklung einzelner Projektschritte

Beispiele für verwendete Prompts:

| Zweck | Beispielhafter Prompt |
|---|---|
| Erklärung von Konzepten | „Erkläre mir einfach, was eine Confusion Matrix bei unserem Klassifikationsmodell aussagt.“ |
| Bericht überarbeiten | „Formuliere den theoretischen Hintergrund wissenschaftlicher.“ |
| Programmierhilfe | „Prüfe das Webcam-Skript und erkläre, warum bestimmte Klassen häufig verwechselt werden.“ |
| Reproduzierbarkeit | „Erstelle einen Prompt, damit das finale Modell mit relativen Pfaden im Repository nutzbar ist.“ |
| README | „Aktualisiere die README auf den aktuellen Projektstand und.“ |

Die KI-Ausgaben wurden geprüft, angepasst und nicht ungeprüft übernommen. Insbesondere gemessene Ergebnisse wie Accuracy, Precision, Recall, F1-Score und Confusion Matrix stammen aus den eigenen Projektläufen und wurden nicht durch KI erzeugt.

## 11. Erklärung zur Arbeitsteilung in der Gruppe

Die Projektarbeit wurde von Ervin Omerovic (209641) und Delil Yilmaz (204114) gemeinsam durchgeführt. Beide Gruppenmitglieder waren an der Themenfindung, der Abstimmung des Projektumfangs, der Diskussion der Ergebnisse sowie an der finalen Durchsicht des Projekts beteiligt.

Ervin Omerovic übernahm schwerpunktmäßig die technische Umsetzung der Datenpipeline, die Organisation der Projektstruktur, die Integration der Datensätze, die Erweiterung der Datenbasis durch eigene Webcam-Aufnahmen, die Durchführung mehrerer Modelltrainings, die Evaluation der Ergebnisse sowie die Einbindung der Webcam-Anwendung. Zusätzlich koordinierte er die Aktualisierung des Berichts, der README und der finalen Abgabeversion des Repositorys.

Delil Yilmaz unterstützte sowohl fachlich als auch technisch bei der Umsetzung des Projekts. Er wirkte bei der Recherche und Auswahl geeigneter Datensätze, der Prüfung der Projektanforderungen sowie der Vorbereitung der Datenstruktur mit. Darüber hinaus unterstützte er bei der Durchführung und Kontrolle einzelner Skripte zur Datenaufbereitung, bei der Überprüfung der erzeugten Ausgaben wie Trainingskurven und Confusion Matrix sowie bei der gemeinsamen Bewertung der Fehlklassifikationen. Zusätzlich beteiligte er sich an der inhaltlichen Kontrolle des Berichts und an der Einordnung der praktischen Nutzbarkeit der Webcam-Erkennung.

Beide Gruppenmitglieder haben den finalen Projektstand, den Bericht und die Abgabeversion des Repositorys überprüft.

## 12. Literaturverzeichnis

Côté, P.-O., Nikanjam, A., Ahmed, N., Humeniuk, D., & Khomh, F. (2024). Data cleaning and machine learning: A systematic literature review. *Automated Software Engineering, 31*(2), 54. https://doi.org/10.1007/s10515-024-00453-w

Daroya, R., Peralta, D., & Naval, P. (2018). Alphabet sign language image classification using deep learning. In *TENCON 2018 – 2018 IEEE Region 10 Conference* (S. 0646–0650). IEEE. https://ieeexplore.ieee.org/abstract/document/8650241/

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*(11), 2278–2324.

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data, 6*(1), 60. https://doi.org/10.1186/s40537-019-0197-0

Tan, C., Sun, F., Kong, T., Zhang, W., Yang, C., & Liu, C. (2018). A survey on deep transfer learning (arXiv:1808.01974). *arXiv*. https://doi.org/10.48550/arXiv.1808.01974

**Letzte Aktualisierung:** 07. Juni 2026
