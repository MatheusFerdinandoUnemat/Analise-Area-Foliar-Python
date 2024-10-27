import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('s4_45dae.jpeg')

# Verificar se a imagem foi carregada corretamente
if imagem is None:
    print("Erro ao carregar a imagem.")
    exit()

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

    # Criar uma nova máscara para objetos que NÃO são brancos (os objetos sobre a folha A4)
    mascara_objetos = cv2.bitwise_not(mascara_branco)

    # Criar uma máscara de zeros (imagem preta) com o mesmo tamanho da imagem original
    mascara_limite = np.zeros_like(mascara_objetos)

    # Preencher a área da folha A4 com branco (255) na máscara limite
    cv2.drawContours(mascara_limite, [contorno_a4], -1, 255, thickness=cv2.FILLED)

    # Aplicar a máscara para limitar a detecção de objetos apenas dentro da folha A4
    mascara_objetos_dentro_a4 = cv2.bitwise_and(mascara_objetos, mascara_limite)

    # Definir cores e limites de segmentação
    cores_segmentos = [
        ((0, 255, 0), np.array([30, 40, 40]), np.array([90, 255, 255])),  # Verde
        ((0, 165, 255), np.array([10, 30, 20]), np.array([20, 255, 200])),  # Marrom
        ((0, 0, 0), np.array([15, 100, 100]), np.array([35, 255, 255])),  # Preto
        ((100, 100, 100), np.array([0, 0, 30]), np.array([180, 20, 80])),  # Cinza escuro
        ((150, 150, 150), np.array([0, 0, 60]), np.array([180, 10, 200])),  # Cinza claro
    ]

    # Lista para armazenar as áreas e posições dos objetos
    lista_areas = []
    y_texto = 50  # Posição inicial da lista de áreas à direita
    area_total_verde_preto = 0  # Variável para armazenar a soma das áreas verde e preto

    # Definir o tamanho da nova área de exibição à direita da imagem (onde a tabela será exibida)
    largura_nova_area = 600
    altura_imagem = imagem.shape[0]
    nova_area = np.ones((altura_imagem, largura_nova_area, 3), dtype=np.uint8) * 255  # Fundo branco

    # Máscaras para armazenar as áreas verdes e pretas
    mascara_verde = np.zeros_like(mascara_objetos_dentro_a4)
    mascara_preto = np.zeros_like(mascara_objetos_dentro_a4)

    # Loop para processar cada segmento
    for cor, limite_inferior, limite_superior in cores_segmentos:
        # Criar máscara para o intervalo de cor correspondente
        mascara_segmento = cv2.inRange(hsv, limite_inferior, limite_superior)
        mascara_segmento = cv2.bitwise_and(mascara_segmento, mascara_objetos_dentro_a4)

        # Encontrar contornos dos objetos segmentados
        contornos_segmento, _ = cv2.findContours(mascara_segmento, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Processar cada contorno encontrado
        for contorno in contornos_segmento:
            if cv2.contourArea(contorno) > 50:  # Limite para evitar ruídos muito pequenos
                # Armazenar máscara verde e preta
                if cor == (0, 255, 0):
                    cv2.drawContours(mascara_verde, [contorno], -1, 255, thickness=cv2.FILLED)  # Preencher a máscara verde
                elif cor == (0, 0, 0):
                    cv2.drawContours(mascara_preto, [contorno], -1, 255, thickness=cv2.FILLED)  # Preencher a máscara preta

    # Combinar as máscaras verde e preta
    mascara_combinada = cv2.bitwise_or(mascara_verde, mascara_preto)

    # Encontrar contornos da máscara combinada
    contornos_combinados, _ = cv2.findContours(mascara_combinada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Processar os contornos combinados
    for contorno in contornos_combinados:
        if cv2.contourArea(contorno) > 50:  # Limite para evitar ruídos muito pequenos
            # Desenhar o contorno combinado na imagem (como verde)
            cv2.drawContours(imagem, [contorno], -1, (0, 255, 0), 2)  # Desenhar como verde

            # Encontrar o centro do contorno
            M = cv2.moments(contorno)
            if M['m00'] != 0:
                centro_x = int(M['m10'] / M['m00'])
                centro_y = int(M['m01'] / M['m00'])

                # Calcular a área
                area_objeto_pixels = cv2.contourArea(contorno)  # Área em pixels
                area_objeto_mm2 = (area_objeto_pixels / cv2.contourArea(contorno_a4)) * (210 * 297)  # Converter para mm²

                # Adicionar a área à soma total caso seja verde ou preto
                area_total_verde_preto += area_objeto_mm2

                # Adicionar o centro e a área à lista
                lista_areas.append((area_objeto_mm2, centro_x, centro_y, (0, 255, 0)))  # Considera apenas como verde

    # Ordenar a lista de áreas detectadas com base na posição vertical (y)
    lista_areas.sort(key=lambda x: x[2])  # Ordena por centro_y, que é o terceiro elemento da tupla (área, centro_x, centro_y, cor)

    # Redefinir a posição Y inicial para exibir na nova área (à direita)
    y_texto = 50

    # Loop para desenhar as setas e escrever as áreas ordenadas
    for area_objeto_mm2, centro_x, centro_y, cor in lista_areas:
        # Desenhar a seta até o lado direito da imagem
        cv2.arrowedLine(imagem, (centro_x, centro_y), (imagem.shape[1] + 50, y_texto), (0, 0, 0), 2, tipLength=0.05)  # Cor preta

        # Adicionar a área à nova área à direita com a mesma cor (preto)
        cv2.putText(nova_area, f'Area: {area_objeto_mm2:.2f} mm2', (20, y_texto),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        # Atualizar a posição Y para a próxima área
        y_texto += 50

    # Adicionar a soma das áreas ao final da tabela
    cv2.putText(nova_area, f'Area foliar total: {area_total_verde_preto:.2f} mm2', (20, y_texto),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Combinar a imagem original com a nova área branca à direita
    imagem_expandida = np.hstack((imagem, nova_area))

    # Redimensionar a imagem expandida para 800x600
    imagem_expandida = cv2.resize(imagem_expandida, (800, 600))

    # Exibir a imagem com as áreas e a tabela
    cv2.imshow('Resultado', imagem_expandida)

    # Aguarda uma tecla para fechar a imagem
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nenhum contorno encontrado para a folha A4.")
