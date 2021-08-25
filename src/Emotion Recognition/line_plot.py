
def chart(emotion[]):
    global x1, a, b
    y1.append(emotion[0])
    y2.append(emotion[1])
    y3.append(emotion[2])
    y4.append(emotion[3])
    y5.append(emotion[4])
    y6.append(emotion[5])
    y7.append(emotion[6])
    x1 += 1
    x.append(x1)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    if (len(y) > 10 ):
        y.pop(0)
        q.pop(0)
        x.pop(0)
        a += 1
        b += 1
    new = curve(x, y)
    ax.plot(new[0],new[1], color='red', linewidth=1, label='training')  # draw line chart
    ax.plot(x,q, color='green', linewidth=1, label='test')

    ax.axis([b, a, 0, 1])
    ax.legend()
    ax.set_xlabel('iteration times')
    ax.set_ylabel('rate')

    # save chart as image
    # fig1 = plt.gcf()
    # plt.close()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    ax.remove()
    buf.seek(0)
    file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    chart = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    chart = cv2.resize(chart, (480, 240))
    if (len(facetem) > 0):
        facecrop1 = cv2.resize(facetem[0], (100, 100))
        newface = cv2.copyMakeBorder(facecrop1, 70, 70, 0, 0, cv2.BORER_CONSTANT, value=[255, 255, 255])
        img1 = cv2.hconcat([newface, chart])D
        _, jpeg1 = cv2.imencode('.jpg', img1)
    else:
        _, jpeg1 = cv2.imencode('.jpg', chart)
    # return HttpResponse(jpeg1.tobytes(),content_type="image/png")

    return jpeg1.tobytes()