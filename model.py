from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Input, Flatten, Dense, Reshape, add
from tensorflow.keras.models import Model

def create_wpod_net_model(input_shape=(None, None, 3)):
    """
    Creates the WPOD-NET model architecture
    This is the model used for license plate detection in the ANPR system
    """
    input_layer = Input(shape=input_shape)
    
    # First block
    x = Conv2D(16, (3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second block
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Third block
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Fourth block
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Fifth block
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Sixth block - detection head
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Output layer - 8 points for the plate corners (x,y) + 1 for confidence
    output = Conv2D(9, (1, 1), padding='same', activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    return model
