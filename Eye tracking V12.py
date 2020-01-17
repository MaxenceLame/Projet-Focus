import numpy as np
import cv2, time
import matplotlib
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('Vid5.mp4')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') #poids entraînés à la détection d'oeil
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)

ret,img= cap.read() 



while cap: 

 ############ pupil detection
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # passage en nuances de gris
	grays = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # passage en nuances de gris

	eye = eye_cascade.detectMultiScale(grays, 1.1, 1) # import des poids entrainés de haar cascade

	exterieur = np.zeros((img.shape[0],img.shape[1]), np.int8) # image vide pour simuler la caméra exterieure


	min=255
	####### Recherche de la valeur la plus faible de l'image
	for i in range(0,gray.shape[0]):
		for j in range(0,gray.shape[1]):
			k=gray[i,j]
			if k<min and k>15:
				min = k
	##############################
	seuil = min+2 # le seuil est légèrement supérieur à cette valeur
	
	retval, thresholded = cv2.threshold(gray, seuil, 255, 0) #On applique ce seuil
    
	closed = cv2.erode(thresholded, kernel, iterations=2) #Fermeture de l'image pour effacer les mauvaise détections de petits éléments

	thresholded, contours, hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) #Détection des contours
	drawing = np.copy(img)
	for contour in contours:

		area = cv2.contourArea(contour) #Transformation des contours en surfaces

		bounding_box = cv2.boundingRect(contour)
		extend = area / (bounding_box[2] * bounding_box[3])
        
		########################## DISCRIMINATION
		if extend > 0.8: #discrimination des surfaces trrop grandes
			continue

		if area < 300: #discrimination des surfaces trrop petites
			continue
		if area >0:
            #discrimine les surfaces qui ne sont pas assez circulaires
			circumference = cv2.arcLength(contour,True)
			circularity = circumference ** 2 / (4*3.14*area)
            # 2 cas de détection supplementaires : soit grosse tache sombre longue / soit petites taches plus circulaires que la pupille
			if len(contours) > 2:
				if circularity > 1.35: # les erreurs ne sont jamais circulaires.         Pour un rond parfait : circularity == 1
					if area < 300: # discrimine les petites taches
						continue
					elif circularity > 2: # discrimine les taches non circulaires
						continue
        ######################################### Normalement, il ne reste que la pupille

        #################  Determination du centre de la pupille, Moments centrés d'ordre 1
		m = cv2.moments(contour)
		if m['m00'] != 0:
			center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])) #Point central de la pupille
		cv2.circle(img, center, 10, (255, 255, 255), -1)
		##############################################################################


		for (x,y,w,h) in eye: # La detection de l'oeil (Haar cascade) permet de savoir quand l'oeil est fermé

			##### ces valeurs de boundig box de l'oeil sont détermées experimentalement
			X=70
			Y=40
			W=300
			H=180
			##############################


			if w>150: #La méthode Haar Cascade met en évidence plusieurs Region of Interest. On ne garde que la plus grande : celle de l'oeil (w>150)
				#cv2.rectangle(img,(X,Y),(X+W,Y+H),(0,0,255),2)

				if w!=0 and h!=0: # Que lorsqu'il y a une détection (lorsque l'oeil est ouvert)

					Xcoeff = 1-((center[0]-X)/W)# Coefficient [0-1] correspondant à la position de la pupille dans la RoI


					###################### prise en compte de la courbe de l'oeil
					if Xcoeff<=0.5:
						Xcoeff = 2.5*Xcoeff*Xcoeff
					else:
						Xcoeff = 1.5*Xcoeff
					############################################@


					Ycoeff = (center[1]-Y)/H# Coefficient [0-1] correspondant à la position de la pupille dans la RoI



					CenterExt =(int(  Xcoeff*exterieur.shape[0]  ),int( (Ycoeff*exterieur.shape[1]) ) )## Position de la vision sur l'environnement extérieur
					cv2.circle(exterieur, CenterExt, 10, (255, 255, 255), -1) # affichage de cette position
 ############################

	
		
	
############## AFFICHAGE ####
	cv2.imshow("interieur", img)
	cv2.imshow("exterieur", exterieur)
	success,img= cap.read()
 ############################

	key =cv2.waitKey(1)
	if key == ord('q'):
		break

cap.release()	
out.release()
cv2.destroyAllWindows()