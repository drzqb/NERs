import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def visualize_single(modelname, historypath):
    '''
    单模型训练可视化
    :param modelname: 模型名称
    :param historypath: 训练指标保存路径
    :return:
    '''

    with open(historypath, "r", encoding="utf-8") as fr:
        history = fr.read()
        history = eval(history)

    plt.subplot(221)
    plt.plot(history["loss"])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')

    plt.subplot(222)
    plt.plot(history["crf_acc"])
    plt.title('acc')
    plt.ylabel('acc')
    plt.xlabel('Epoch')

    plt.subplot(223)
    plt.plot(history["val_loss"])
    plt.title('val_loss')
    plt.ylabel('val_loss')
    plt.xlabel('Epoch')

    plt.subplot(224)
    plt.plot(history["val_crf_acc"])
    plt.title('val_acc')
    plt.ylabel('val_acc')
    plt.xlabel('Epoch')

    plt.suptitle("Model Metrics")

    plt.tight_layout()
    plt.savefig(modelname + ".jpg", dpi=500, bbox_inches="tight")


def visualize_loss_acc():
    '''
    多模型对比
    :return:
    '''
    with open("model/medical_bertcrf/history.txt", "r", encoding="utf-8") as fr:
        history_bertcrf = fr.read()
        history_bertcrf = eval(history_bertcrf)

    with open("model/medical_bertbigrucrf/history.txt", "r", encoding="utf-8") as fr:
        history_bertbigrucrf = fr.read()
        history_bertbigrucrf = eval(history_bertbigrucrf)

    with open("model/medical_bertlinear/history.txt", "r", encoding="utf-8") as fr:
        history_bertlinear = fr.read()
        history_bertlinear = eval(history_bertlinear)

    with open("model/medical_robertcrf/history.txt", "r", encoding="utf-8") as fr:
        history_robertcrf = fr.read()
        history_robertcrf = eval(history_robertcrf)

    with open("model/medical_robertbigrucrf/history.txt", "r", encoding="utf-8") as fr:
        history_robertbigrucrf = fr.read()
        history_robertbigrucrf = eval(history_robertbigrucrf)

    with open("model/medical_robertlinear/history.txt", "r", encoding="utf-8") as fr:
        history_robertlinear = fr.read()
        history_robertlinear = eval(history_robertlinear)

    plt.subplot(221)
    plt.plot(history_bertcrf["loss"])
    plt.plot(history_bertbigrucrf["loss"])
    plt.plot(history_bertlinear["loss"])
    plt.plot(history_robertcrf["loss"])
    plt.plot(history_robertbigrucrf["loss"])
    plt.plot(history_robertlinear["loss"])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['CRF', 'BIGRUCRF', 'Linear', 'roCRF', 'roGRUCRF', 'roLinear'], loc='best', prop={'size': 4})

    plt.subplot(222)
    plt.plot(history_bertcrf["crf_acc"])
    plt.plot(history_bertbigrucrf["crf_acc"])
    plt.plot(history_bertlinear["acc"])
    plt.plot(history_robertcrf["crf_acc"])
    plt.plot(history_robertbigrucrf["crf_acc"])
    plt.plot(history_robertlinear["acc"])
    plt.title('acc')
    plt.ylabel('acc')
    plt.xlabel('Epoch')
    plt.legend(['CRF', 'BIGRUCRF', 'Linear', 'roCRF', 'roGRUCRF', 'roLinear'], loc='best', prop={'size': 4})

    plt.subplot(223)
    plt.plot(history_bertcrf["val_loss"])
    plt.plot(history_bertbigrucrf["val_loss"])
    plt.plot(history_bertlinear["val_loss"])
    plt.plot(history_robertcrf["val_loss"])
    plt.plot(history_robertbigrucrf["val_loss"])
    plt.plot(history_robertlinear["val_loss"])
    plt.title('val_loss')
    plt.ylabel('val_loss')
    plt.xlabel('Epoch')
    plt.legend(['CRF', 'BIGRUCRF', 'Linear', 'roCRF', 'roGRUCRF', 'roLinear'], loc='best', prop={'size': 4})

    plt.subplot(224)
    plt.plot(history_bertcrf["val_crf_acc"])
    plt.plot(history_bertbigrucrf["val_crf_acc"])
    plt.plot(history_bertlinear["val_acc"])
    plt.plot(history_robertcrf["val_crf_acc"])
    plt.plot(history_robertbigrucrf["val_crf_acc"])
    plt.plot(history_robertlinear["val_acc"])
    plt.title('val_acc')
    plt.ylabel('val_acc')
    plt.xlabel('Epoch')
    plt.legend(['CRF', 'BIGRUCRF', 'Linear', 'roCRF', 'roGRUCRF', 'roLinear'], loc='best', prop={'size': 4})

    plt.suptitle("Model Metrics")

    plt.tight_layout()
    plt.savefig("compare.jpg", dpi=500, bbox_inches="tight")

    # plt.show()


def visualize_PRF():
    with open("model/mrc_span_tape/history.txt", "r", encoding="utf-8") as fr:
        history_tape = fr.read()
        history_tape = eval(history_tape)

    gs = gridspec.GridSpec(2, 6)
    plt.subplot(gs[0, 1:3])
    plt.plot(history_tape["loss"])
    plt.plot(history_tape["val_loss"])
    plt.grid()
    plt.title('loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='best', prop={'size': 4})

    plt.subplot(gs[0, 3:5])
    plt.plot(history_tape["acc"])
    plt.plot(history_tape["val_acc"])
    plt.grid()
    plt.title('acc')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, :2])
    plt.plot(history_tape["precision"])
    plt.plot(history_tape["val_precision"])
    plt.grid()
    plt.title('precision')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 2:4])
    plt.plot(history_tape["recall"])
    plt.plot(history_tape["val_recall"])
    plt.grid()
    plt.title('recall')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 4:])
    plt.plot(history_tape["f1"])
    plt.plot(history_tape["val_f1"])
    plt.grid()
    plt.title('f1')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='best', prop={'size': 4})

    plt.suptitle("Model Metrics")

    plt.tight_layout()
    plt.savefig("mrc_span_tape_PRF.jpg", dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    # visualize_PRF()
    visualize_single("grucrfe", "model/medical_grucrfe/history.txt")
