####vanilla gan
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output) ## 이것이 실제 데이터라는것을 학습시키기 위한
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output) ## discriminator를 속이기 위한
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)  ## discriminator를 속이기 위해  tf.ones_like 를 사용하여 가짜 샘플을 실제 데이터라며 들임.?(



####wgan
def critic_loss(r_logit, f_logit):
    real_loss = - tf.reduce_mean(r_logit)
    fake_loss = tf.reduce_mean(f_logit)
    return real_loss, fake_loss

def generator_loss(f_logit):
    fake_loss = - tf.reduce_mean(f_logit)
    return fake_loss

generator_optimizer = tf.keras.optimizers.Adam(learning_rate= LR) #tf.keras.optimizers.RMSprop(learning_rate= LR)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate= LR) #tf.keras.optimizers.RMSprop(learning_rate= LR)

