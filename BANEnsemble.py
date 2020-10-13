import tensorflow as tf
class BANEnsemble(tf.keras.models.Model):
  def __init__(self, students, **kwargs):
    super(BANEnsemble, self).__init__(**kwargs)
    self.students = students
    self.len = len(students)
  
  def call(self, inputs):
    s_out = []
    for student in self.students:
      s_out.append(student(inputs))
      
    
    x = tf.keras.layers.Add()(s_out)
    x = tf.keras.layers.Lambda(lambda y: y/self.len)(x)
    return x

