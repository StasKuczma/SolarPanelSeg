# Podsumowanie Modelu Segmentacji Paneli Fotowoltaicznych

## Do czego służy wytrenowany model?
Model został wytrenowany do segmentacji paneli fotowoltaicznych na dachach budynków jednorodzinnych. Jego zadaniem jest wykrywanie obszarów pokrytych panelami oraz rozróżnianie ich od tła.

## Na jakich danych został wytrenowany model?
Model został wytrenowany na dwóch zestawach danych:
1. Publicznie dostępny zbiór danych: [Zenodo - Solar Panel Segmentation](https://zenodo.org/records/7358126).
2. Własne dane przygotowane w oparciu o precyzyjną ortofotomapę Poznania.

## Dla jakiej rozdzielczości przestrzennej został wytrenowany model?
Model został wytrenowany na obrazach o rozdzielczości 512x512 pikseli.

## Na jakim źródle danych model sprawdza się najlepiej?
Model osiąga najlepsze wyniki na obrazach z ortofotomap o wysokiej jakości, zwłaszcza w obszarach miejskich, gdzie panele fotowoltaiczne są dobrze widoczne i kontrastują z otoczeniem.

## Przykłady najlepszych i najgorszych rezultatów
**Najlepsze rezultaty:**
- Obrazy o wysokim kontraście i dobrym oświetleniu.
- Panele fotowoltaiczne o wyraźnych krawędziach i jednolitym kolorze.
  ![screenshot_from_2025-01-30_12-49-43_720](https://github.com/user-attachments/assets/8d6ef4a9-41d3-4fe6-8540-78dda311ca12)


**Najgorsze rezultaty:**
- Obszary zacienione lub częściowo zasłonięte.
- Dachy z panelami o nieregularnym układzie lub w złym stanie technicznym.
- Obrazy o niskim kontraście lub dużym poziomie szumów.
![Screenshot from 2025-01-29 14-21-06](https://github.com/user-attachments/assets/ee02fe5a-abf0-4e01-b06c-599d4969544a)
![screenshot_from_2025-01-30_12-49-19_720](https://github.com/user-attachments/assets/97945365-c580-487d-9e53-2ed356941d4d)

## Metryki działania modelu
- **IoU - Intersection over Union:** osiągało wartość ponad 0.8:
- 
  ![IoU](https://github.com/user-attachments/assets/15464f5b-b634-4253-8338-ce08ca4c43c5)
  
- **Validation loss**:
- 
![Validation_loss](https://github.com/user-attachments/assets/77dc2ce8-af16-4fd8-aa8d-a7bb4ad77fa5)






run container `./docker/run_gpu.sh`
