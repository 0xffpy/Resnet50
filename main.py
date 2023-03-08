def main():

    image_size = (224,224)
    train_dataset = image_dataset_from_directory(
        data_dir,
        batch_size=32,
        image_size=image_size,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='training'
    )
    val_dataset = image_dataset_from_directory(
        data_dir,
        batch_size=32,
        image_size=image_size,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='validation'
    )

    class_names = train_dataset.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    model = Model_ResNet50(image_size+(3,))
    
    model.compile(optimizer='adam',
             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
    
    history = model.fit(train_dataset, 
          validation_data=val_dataset,
         epochs=100)
    
    df_loss_acc = pd.DataFrame(history.history)
    df_loss= df_loss_acc[['loss','val_loss']]
    df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
    df_acc= df_loss_acc[['accuracy','val_accuracy']]
    df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
    df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
    df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
