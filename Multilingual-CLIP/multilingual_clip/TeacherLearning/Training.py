import Dataset, TrainingModel
import tensorflow as tf
import transformers
import datasets
import Utils


def loadTextTranslations():
    return datasets.load_dataset('Arabic-Clip/ImageCaptions-7M-Translations-Arabic')['train']


def loadTargetEmbeddings(imageBase="Vit-B-32", validationSize=5000):
    trainSamples = datasets.load_dataset('M-CLIP/ImageCaptions-7M-Embeddings', imageBase,
                                         split='train[{}:]'.format(validationSize))
    valSamples = datasets.load_dataset('M-CLIP/ImageCaptions-7M-Embeddings', imageBase,
                                       split='train[:{}]'.format(validationSize))

    embeddingShape = tf.convert_to_tensor(trainSamples[0]['embedding']).shape
    return trainSamples, valSamples, embeddingShape


def singleGPUTraining():
    numValidationSamples = 5000
    stepsPerEpoch, lr = 1000, 0.00001
    gradAccumSteps, batchSize = 1, 256
    numTrainSteps, numWarmupSteps = 99999999, 1000

    modelBase = 'aubmindlab/bert-base-arabertv2'
    tokenizerBase = 'aubmindlab/bert-base-arabertv2'
    imageBase = "Vit-B-32"
    modelName = '{}-{}'.format(modelBase, imageBase)

    startWeights = "/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/aubmindlab/bert-base-arabertv2-Vit-B-32.index"
    targetCaptions = loadTextTranslations()
    trainEmbeddings, valEmbeddings, imageEncoderDimensions = loadTargetEmbeddings(validationSize=numValidationSamples)

    def createOptimizerFunc():
        optimizer, schedule = transformers.optimization_tf.create_optimizer(lr, numTrainSteps, numWarmupSteps)
        if (gradAccumSteps <= 1):
            return optimizer
        else:
            return Utils.GradientAccumulator(optimizer, gradAccumSteps)

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizerBase)
    print("="*100)
    print("imageEncoderDimensions[-1]: ", imageEncoderDimensions[-1])
    print("="*100)

    model = TrainingModel.SentenceModelWithLinearTransformation(modelBase, imageEncoderDimensions[-1])

    if (startWeights is not None):
        model.load_weights(startWeights)
    model.compile(createOptimizerFunc(), 'mse', metrics=['mae', 'cosine_similarity'])

    trainDataset, valDataset = Dataset.createTrainingAndValidationDataset(trainEmbeddings, valEmbeddings, batchSize,
                                                                          tokenizer,
                                                                          targetCaptions=targetCaptions,
                                                                          encoderDims=imageEncoderDimensions)

    if (gradAccumSteps > 1):  # In order to make fair logging on Wandb
        stepsPerEpoch *= gradAccumSteps

    model.fit(trainDataset, epochs=1000, steps_per_epoch=stepsPerEpoch,
              validation_data=valDataset,
              callbacks=[
                  Utils.CustomSaveCallBack(modelName, saveInterval=5, firstSavePoint=5),
              ]
              )
    # self.model.save_weights(self.saveName.format(epoch + 1))

if __name__ == '__main__':
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        singleGPUTraining()