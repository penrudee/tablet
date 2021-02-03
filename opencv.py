origin_img = cv2.imread(IMG)

        img = image_resize(origin_img, height=300) 
        # recbox, label, configuration = _cvlib.detect_common_objects(img)
        # output_image = draw_bbox(img, recbox, label, configuration)
        
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        imgGry = cv2.cvtColor(hsv_img,cv2.COLOR_BGR2GRAY)
        Gry32 = numpy.float32(imgGry)
        # Histogram Equalization https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
        equ = cv2.equalizeHist(imgGry)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        cl1 = clahe.apply(imgGry) #เทคนิค  histogram equa อีกอย่าง
        img_with_edge = cv2.Canny(cl1,150,255) #สร้างกรอบบนรูปที่เราต้องการ
        dst = cv2.cornerHarris(img_with_edge, 8, 3, 0.04) #ตรวจจับมุม
        dst = cv2.dilate(dst,None)
        img[dst>0.02*dst.max()] = [255, 0 ,255]
        res = numpy.hstack((imgGry,equ,cl1,img_with_edge)) #แสดงภาพ 3 ภาพเรียงซ้อนกันจากซ้ายไปขวา

        blurred = cv2.GaussianBlur(cl1,(5,5),0)
        thresh = cv2.threshold(blurred, 95, 255, cv2.THRESH_BINARY)[1]
        
        
        ksize = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize,ksize))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  
        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] != 0:  # prevent divide by zero
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))
                approx = cv2.approxPolyDP(c, 0.03*cv2.arcLength(c, True), True)
                cv2.drawContours(img, [approx], 0, (0), 5)
                x = approx.ravel()[0]
                y = approx.ravel()[1]
                if len(approx) == 3:
                    cv2.putText(img, "Triangle", (x, y), font, 1, (0, 255, 0))
                    shape_name = 'triangle'
                elif len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspectRatio = float(w)/h
                    print(f'aspect ratio ={aspectRatio}')
                    if aspectRatio >= 0.95 and aspectRatio < 1.05:
                        cv2.putText(img, "Rectangle", (x, y), font, 1, (0, 0, 255))
                        shape_name = 'rectangle'
                        shapes.append(shape_name)

                elif len(approx) == 5:
                    cv2.putText(img, "Pentagon", (x, y), font, 1, (255, 100, 0))
                    shape_name = 'Pentagon'
                    shapes.append(shape_name)
                elif len(approx) == 6:
                    cv2.putText(img, "Hexagon", (x, y), font, 1, (29, 191, 66))
                    shape_name = 'Hexagon'
                    shapes.append(shape_name)
                elif len(approx) >= 11:
                    cv2.putText(img, "Oval", (x, y), font, 1, (100, 100, 100))
                    shape_name = 'Oval'
                    shapes.append(shape_name)

                elif 6 < len(approx) < 15:
                    cv2.putText(img, "Ellipse", (x, y), font, 1, (10, 100, 200))
                    shape_name = 'Ellipse'
                    shapes.append(shape_name)

                else:
                    cv2.putText(img, "Circle", (x, y), font, 1, (29, 191, 66))
                    shape_name = 'circle'
                    shapes.append(shape_name)
            # c = c.astype("float")
            # c = c.astype("int")

            # cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
            
            
        
        print(f'shapes lis ={shapes}')        
        cv2.imshow("3image", res)
        cv2.imshow("shape",img)
        cv2.imshow("Threshold", thresh)
        # cv2.imshow("img",output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
