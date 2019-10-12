#Calculate loss
def compute_loss(AL, Y):
    if args.loss == "ce":#crossentropy loss
        m = Y.shape[1]
        loss = -1 * np.sum((Y * np.log(AL)))
        loss = np.squeeze(loss)
        return loss
    else:
        # Squared Error loss
        return ((AL - Y) ** 2).mean() #al indicates last layer y indicates labels
