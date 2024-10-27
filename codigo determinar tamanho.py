import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('s4C_45dae.jpeg')

# Converter a imagem para o espaço de cores HSV
hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

# Definir o intervalo para detectar a cor branca (incluindo tons acinzentados)
limite_inferior_branco = np.array([0, 0, 150])  # Branco
limite_superior_branco = np.array([180, 30, 255])  # Branco

# Criar uma máscara para a cor branca (folha A4)
mascara_branco = cv2.inRange(hsv, limite_inferior_branco, limite_superior_branco)

# Encontrar contornos na máscara (para identificar a folha A4)
contornos_brancos, _ = cv2.findContours(mascara_branco, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assumindo que o maior contorno branco é a folha A4
if contornos_brancos:  # Verifica se há contornos encontrados
    contorno_a4 = max(contornos_brancos, key=cv2.contourArea)

    # Desenhar o contorno da folha A4 na imagem
    cv2.drawContours(imagem, [contorno_a4], -1, (255, 0, 0), 2)  # Contorno azul para a folha A4

    # Calcular a altura e a largura da folha A4 em pixels
    x, y, largura_a4, altura_a4 = cv2.boundingRect(contorno_a4)

    # A folha A4 tem 210 mm x 297 mm. Vamos calcular a relação de pixels por milímetro.
    mm_por_pixel_largura = 210 / largura_a4  # 210 mm para a largura da folha A4
    mm_por_pixel_altura = 297 / altura_a4    # 297 mm para a altura da folha A4

    # Desenhar linhas verticais com espaçamento de 1 cm (10 mm)
    espacamento_vertical = int(10 / mm_por_pixel_largura)  # Converter 1 cm para pixels
    for i, indice_vertical in enumerate(range(x, x + largura_a4, espacamento_vertical), start=1):
        cv2.line(imagem, (indice_vertical, y), (indice_vertical, y + altura_a4), (0, 255, 0), 1)
        # Adicionar numeração na parte superior da imagem
        cv2.putText(imagem, str(i), (indice_vertical, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Desenhar linhas horizontais com espaçamento de 1 cm (10 mm)
    espacamento_horizontal = int(10 / mm_por_pixel_altura)  # Converter 1 cm para pixels
    for j, indice_horizontal in enumerate(range(y, y + altura_a4, espacamento_horizontal), start=1):
        cv2.line(imagem, (x, indice_horizontal), (x + largura_a4, indice_horizontal), (0, 255, 0), 1)
        # Adicionar numeração na parte esquerda da imagem
        cv2.putText(imagem, str(j), (x - 20, indice_horizontal), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

# Redimensionar a imagem para o tamanho desejado
largura_desejada = 800
altura_desejada = 600
imagem_redimensionada = cv2.resize(imagem, (largura_desejada, altura_desejada))

# Exibir a imagem com a folha A4, a grade de 1 cm e as numerações
cv2.imshow('Folha A4 com Grade de 1 cm e Numeração', imagem_redimensionada)
cv2.waitKey(0)
cv2.destroyAllWindows()
