

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class ChannelAttention(layers.Layer):
    
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.shared_dense_1 = layers.Dense(
            channels // self.ratio,
            activation='relu',
            name='channel_att_fc1'
        )
        self.shared_dense_2 = layers.Dense(
            channels,
            name='channel_att_fc2'
        )
        super().build(input_shape)
    
    def call(self, x):
        avg_pool = layers.GlobalAveragePooling3D(keepdims=True)(x)
        avg_pool = self.shared_dense_1(avg_pool)
        avg_pool = self.shared_dense_2(avg_pool)
        
        max_pool = tf.reduce_max(x, axis=[1, 2, 3], keepdims=True)
        max_pool = self.shared_dense_1(max_pool)
        max_pool = self.shared_dense_2(max_pool)
        
        attention = layers.Activation('sigmoid')(
            layers.Add()([avg_pool, max_pool])
        )
        return layers.Multiply()([x, attention])
    
    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


class MultiScaleMotionMagnitudeModule(layers.Layer):
    
    def __init__(self, filters=32, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
    
    def build(self, input_shape):
        self.motion_conv_d1 = layers.Conv3D(
            self.filters // 3, (3, 3, 3),
            padding='same',
            dilation_rate=(1, 1, 1),
            activation='relu',
            name='motion_fine_d1'
        )
        self.motion_conv_d2 = layers.Conv3D(
            self.filters // 3, (3, 3, 3),
            padding='same',
            dilation_rate=(1, 2, 2),
            activation='relu',
            name='motion_medium_d2'
        )
        self.motion_conv_d4 = layers.Conv3D(
            self.filters // 3, (3, 3, 3),
            padding='same',
            dilation_rate=(1, 4, 4),
            activation='relu',
            name='motion_coarse_d4'
        )
        
        self.bn_d1 = layers.BatchNormalization()
        self.bn_d2 = layers.BatchNormalization()
        self.bn_d4 = layers.BatchNormalization()
        
        self.fusion_conv = layers.Conv3D(
            self.filters, (1, 1, 1),
            padding='same',
            activation='relu',
            name='motion_fusion'
        )
        self.fusion_bn = layers.BatchNormalization()
        
        self.motion_magnitude = layers.Conv3D(
            self.filters, (1, 1, 1),
            padding='same',
            activation='sigmoid',
            name='motion_magnitude_gate'
        )
        
        super().build(input_shape)
    
    def call(self, x, training=None):
        motion_signal = tf.concat([
            tf.zeros_like(x[:, 0:1, :, :, :]),
            x[:, 1:, :, :, :] - x[:, :-1, :, :, :]
        ], axis=1)
        
        motion_d1 = self.motion_conv_d1(motion_signal)
        motion_d1 = self.bn_d1(motion_d1, training=training)
        
        motion_d2 = self.motion_conv_d2(motion_signal)
        motion_d2 = self.bn_d2(motion_d2, training=training)
        
        motion_d4 = self.motion_conv_d4(motion_signal)
        motion_d4 = self.bn_d4(motion_d4, training=training)
        
        motion_features = layers.Concatenate(name='concat_motion_scales')([
            motion_d1, motion_d2, motion_d4
        ])
        
        motion_features = self.fusion_conv(motion_features)
        motion_features = self.fusion_bn(motion_features, training=training)
        
        motion_map = self.motion_magnitude(motion_features)
        
        return motion_map
    
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


class MultiScaleMotionDirectionModule(layers.Layer):
    
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
    
    def build(self, input_shape):
        self.conv_h_d1 = layers.Conv3D(
            self.channels // 3, (1, 1, 3),
            padding='same',
            dilation_rate=(1, 1, 1),
            activation='relu',
            name='dir_h_fine'
        )
        self.conv_h_d2 = layers.Conv3D(
            self.channels // 3, (1, 1, 3),
            padding='same',
            dilation_rate=(1, 1, 2),
            activation='relu',
            name='dir_h_medium'
        )
        self.conv_h_d4 = layers.Conv3D(
            self.channels // 3, (1, 1, 3),
            padding='same',
            dilation_rate=(1, 1, 4),
            activation='relu',
            name='dir_h_coarse'
        )
        
        self.conv_w_d1 = layers.Conv3D(
            self.channels // 3, (1, 3, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            activation='relu',
            name='dir_v_fine'
        )
        self.conv_w_d2 = layers.Conv3D(
            self.channels // 3, (1, 3, 1),
            padding='same',
            dilation_rate=(1, 2, 1),
            activation='relu',
            name='dir_v_medium'
        )
        self.conv_w_d4 = layers.Conv3D(
            self.channels // 3, (1, 3, 1),
            padding='same',
            dilation_rate=(1, 4, 1),
            activation='relu',
            name='dir_v_coarse'
        )
        
        self.bn_h = layers.BatchNormalization()
        self.bn_w = layers.BatchNormalization()
        
        self.fusion_h = layers.Conv3D(
            self.channels, (1, 1, 1),
            padding='same',
            activation='relu',
            name='fusion_h'
        )
        self.fusion_w = layers.Conv3D(
            self.channels, (1, 1, 1),
            padding='same',
            activation='relu',
            name='fusion_w'
        )
        
        self.att_h = layers.Conv3D(
            self.channels, (1, 1, 1),
            padding='same',
            activation='sigmoid',
            name='attention_h'
        )
        self.att_w = layers.Conv3D(
            self.channels, (1, 1, 1),
            padding='same',
            activation='sigmoid',
            name='attention_w'
        )
        
        super().build(input_shape)
    
    def call(self, x, training=None):
        h_d1 = self.conv_h_d1(x)
        h_d2 = self.conv_h_d2(x)
        h_d4 = self.conv_h_d4(x)
        
        h = layers.Concatenate(name='concat_h_scales')([h_d1, h_d2, h_d4])
        h = self.fusion_h(h)
        h = self.bn_h(h, training=training)
        
        h_att = self.att_h(h)
        h_weighted = layers.Multiply(name='h_weighted')([h, h_att])
        
        w_d1 = self.conv_w_d1(x)
        w_d2 = self.conv_w_d2(x)
        w_d4 = self.conv_w_d4(x)
        
        w = layers.Concatenate(name='concat_w_scales')([w_d1, w_d2, w_d4])
        w = self.fusion_w(w)
        w = self.bn_w(w, training=training)
        
        w_att = self.att_w(w)
        w_weighted = layers.Multiply(name='w_weighted')([w, w_att])
        
        output = layers.Add(name='combine_directions')([h_weighted, w_weighted])
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config



# MSMANET 

def build_msmanet(
    input_shape=(2, 64, 64, 3),
    output_frames=14,
    filters=[128, 128, 128, 64],
    motion_filters=64,
    initial_filters=64,
    reduction_ratio=8,
    dropout_rate=0.1
):
    
    inp = layers.Input(shape=input_shape, name='input_frames')
    
    x_implicit = layers.Conv3D(initial_filters, (3, 3, 3), padding='same', name='initial_conv')(inp)
    x_implicit = layers.BatchNormalization()(x_implicit)
    x_implicit = layers.Activation('relu')(x_implicit)
    
    n_blocks = len(filters)
    
    for i, filter_count in enumerate(filters):
        residual = x_implicit
        
        branch_d1 = layers.Conv3D(
            filter_count // 3, (3, 3, 3),
            padding='same',
            dilation_rate=(1, 1, 1),
            name=f'implicit_block{i}_d1'
        )(x_implicit)
        branch_d1 = layers.BatchNormalization()(branch_d1)
        branch_d1 = layers.Activation('relu')(branch_d1)
        
        branch_d2 = layers.Conv3D(
            filter_count // 3, (3, 3, 3),
            padding='same',
            dilation_rate=(1, 2, 2),
            name=f'implicit_block{i}_d2'
        )(x_implicit)
        branch_d2 = layers.BatchNormalization()(branch_d2)
        branch_d2 = layers.Activation('relu')(branch_d2)
        
        branch_d4 = layers.Conv3D(
            filter_count // 3, (3, 3, 3),
            padding='same',
            dilation_rate=(1, 4, 4),
            name=f'implicit_block{i}_d4'
        )(x_implicit)
        branch_d4 = layers.BatchNormalization()(branch_d4)
        branch_d4 = layers.Activation('relu')(branch_d4)
        
        x_implicit = layers.Concatenate(name=f'concat_implicit_block{i}')([
            branch_d1, branch_d2, branch_d4
        ])
        
        x_implicit = layers.Conv3D(
            filter_count, (1, 1, 1),
            padding='same',
            name=f'fusion_implicit_block{i}'
        )(x_implicit)
        x_implicit = layers.BatchNormalization()(x_implicit)
        x_implicit = layers.Activation('relu')(x_implicit)
        
        x_implicit = layers.Conv3D(
            filter_count, (3, 3, 3),
            padding='same',
            name=f'refine_implicit_block{i}'
        )(x_implicit)
        x_implicit = layers.BatchNormalization()(x_implicit)
        
        x_implicit = ChannelAttention(
            ratio=reduction_ratio,
            name=f'channel_att_block{i}'
        )(x_implicit)
        
        if residual.shape[-1] != filter_count:
            residual = layers.Conv3D(
                filter_count, (1, 1, 1),
                padding='same',
                name=f'residual_proj_block{i}'
            )(residual)
        
        x_implicit = layers.Add(name=f'residual_add_block{i}')([x_implicit, residual])
        x_implicit = layers.Activation('relu')(x_implicit)
        

        if i < n_blocks - 1:
            x_implicit = layers.SpatialDropout3D(dropout_rate)(x_implicit)
    
    x_feat = layers.Conv3D(
        motion_filters, (3, 3, 3),
        padding='same',
        activation='relu',
        name='motion_feature_extraction'
    )(inp)
    x_feat = layers.BatchNormalization()(x_feat)
    
    magnitude_map = MultiScaleMotionMagnitudeModule(
        filters=motion_filters,
        name='motion_magnitude_module'
    )(inp)
    
    direction_map = MultiScaleMotionDirectionModule(
        channels=motion_filters,
        name='motion_direction_module'
    )(x_feat)
    
    gated = layers.Multiply(name='magnitude_gates_direction')([
        magnitude_map,
        direction_map
    ])
    
    x_motion = layers.Add(name='motion_hybrid_fusion')([gated, x_feat])
    x_motion = layers.Activation('relu')(x_motion)
    
    x = layers.Concatenate(axis=-1, name='concat_implicit_explicit')([
        x_implicit,
        x_motion
    ])
    
    fusion_filters = max(64, motion_filters)
    x = layers.Conv3D(
        fusion_filters, (1, 1, 1),
        padding='same',
        activation='relu',
        name='fusion_conv'
    )(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv3D(
        fusion_filters, (3, 3, 3),
        padding='same',
        activation='relu',
        name='fusion_refine'
    )(x)
    x = layers.BatchNormalization()(x)
    
    output_filters = max(32, fusion_filters // 2)
    x = layers.Conv3D(
        output_filters, (3, 3, 3),
        padding='same',
        activation='relu',
        name='output_conv1'
    )(x)
    
    if fusion_filters > 64:
        x = layers.Conv3D(
            output_filters // 2, (3, 3, 3),
            padding='same',
            activation='relu',
            name='output_conv2'
        )(x)

    x = layers.Conv3DTranspose(
        output_filters,
        kernel_size=(7, 1, 1),
        strides=(7, 1, 1),
        padding='same',
        activation='relu',
        name='temporal_upsample'
    )(x)
    
    output = layers.Conv3D(
        3, (3, 3, 3),
        padding='same',
        activation='sigmoid',
        name='output_prediction'
    )(x)
    
    model = tf.keras.Model(inp, output, name='MSMANet')
    
    return model

#MSMAUNet
def build_msmaunet(
    input_shape=(2, 64, 64, 3),
    output_frames=14,
    filters=[128, 128, 128, 64],
    motion_filters=64,
    initial_filters=64,
    reduction_ratio=8,
    dropout_rate=0.1
):
    
    inp = layers.Input(shape=input_shape, name='input_frames')
    

    x = layers.Conv3D(initial_filters, (3, 3, 3), padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    encoder_features = []  
    
    for i, filter_count in enumerate(filters):
        residual = x
        
        branch_d1 = layers.Conv3D(filter_count // 3, (3, 3, 3), padding='same', dilation_rate=(1, 1, 1))(x)
        branch_d1 = layers.BatchNormalization()(branch_d1)
        branch_d1 = layers.Activation('relu')(branch_d1)
        
        branch_d2 = layers.Conv3D(filter_count // 3, (3, 3, 3), padding='same', dilation_rate=(1, 2, 2))(x)
        branch_d2 = layers.BatchNormalization()(branch_d2)
        branch_d2 = layers.Activation('relu')(branch_d2)
        
        branch_d4 = layers.Conv3D(filter_count // 3, (3, 3, 3), padding='same', dilation_rate=(1, 4, 4))(x)
        branch_d4 = layers.BatchNormalization()(branch_d4)
        branch_d4 = layers.Activation('relu')(branch_d4)
        
        x = layers.Concatenate()([branch_d1, branch_d2, branch_d4])
        x = layers.Conv3D(filter_count, (1, 1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv3D(filter_count, (3, 3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = ChannelAttention(ratio=reduction_ratio)(x)
        
        if residual.shape[-1] != filter_count:
            residual = layers.Conv3D(filter_count, (1, 1, 1), padding='same')(residual)
        
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        
        
        encoder_features.append(x)
        
        
        if i < len(filters) - 1:
            x = layers.Conv3D(filter_count, (2, 2, 2), strides=(1, 2, 2), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.SpatialDropout3D(dropout_rate)(x)
    
    x_feat = layers.Conv3D(motion_filters, (3, 3, 3), padding='same', activation='relu')(inp)
    x_feat = layers.BatchNormalization()(x_feat)
    
    magnitude_map = MultiScaleMotionMagnitudeModule(filters=motion_filters)(inp)
    direction_map = MultiScaleMotionDirectionModule(channels=motion_filters)(x_feat)
    
    gated = layers.Multiply()([magnitude_map, direction_map])
    x_motion = layers.Add()([gated, x_feat])
    x_motion = layers.Activation('relu')(x_motion)
    
   
    for i in range(len(filters) - 1):
        x_motion = layers.MaxPooling3D(pool_size=(1, 2, 2))(x_motion)
    
    x = layers.Concatenate(axis=-1)([x, x_motion])
    x = layers.Conv3D(filters[-1], (1, 1, 1), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters[-1], (3, 3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    decoder = x
    
    for i in range(len(filters) - 1, -1, -1):
        filter_count = filters[i]
        
        if i < len(filters) - 1:
            decoder = layers.Conv3DTranspose(filter_count, (2, 2, 2), strides=(1, 2, 2), padding='same')(decoder)
            decoder = layers.BatchNormalization()(decoder)
            decoder = layers.Activation('relu')(decoder)
        
        skip = encoder_features[i]
        
        decoder_channels = decoder.shape[-1]
        skip_channels = skip.shape[-1]
        
        if decoder_channels != skip_channels:
            decoder = layers.Conv3D(skip_channels, (1, 1, 1), padding='same')(decoder)
            decoder = layers.BatchNormalization()(decoder)
            decoder = layers.Activation('relu')(decoder)
        
        decoder = layers.Concatenate()([decoder, skip])
        
        decoder = layers.Conv3D(filter_count, (3, 3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        
        decoder = ChannelAttention(ratio=reduction_ratio)(decoder)
    
    decoder = layers.Conv3DTranspose(
        64,
        kernel_size=(7, 1, 1),
        strides=(7, 1, 1),
        padding='same',
        activation='relu'
    )(decoder)
    
    output = layers.Conv3D(
        3, (3, 3, 3),
        padding='same',
        activation='sigmoid'
    )(decoder)
    
    model = tf.keras.Model(inp, output, name='MSMAUNet')
    
    return model


if __name__ == '__main__':
    
    model = build_msmanet()
    model_unet = build_msmaunet()

