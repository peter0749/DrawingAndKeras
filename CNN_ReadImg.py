def Read_IMGS(width=320, height=240, dirPos='data'):
    imgList = [f for f in listdir(dirPos) if isfile(join(dirPos,f))]
    imgnum = len(imgList)
    labels = []
    imgs = np.empty((imgnum, 3, height, width))
    i=0
    for path in imgList:
        lab = path.split("-")[2].split(".")[0]
        labels.append(lab)
        img = Image.open(dirPos+'/'+path)
        img = img.resize((height,width), Image.BILINEAR)
        img = np.asarray(img, dtype='float32')
        imgs[i,:,:,:] = img
        i+=1
    #img.show()
    return imgs, labels

