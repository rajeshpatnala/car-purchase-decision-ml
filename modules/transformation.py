import pandas

def ip_transformation(inputs):
    ip_tfm = dict((("more",6), ("5more",6), ("vhigh",4), 
                   ("high",3), ("high",3), ("big",3), 
                   ("med",2), ("small",1), ("low",1)))
    transformed_inputs = []
    for value in inputs:
        tf_val = ip_tfm.get(value, value)
        transformed_inputs.append(int(tf_val))
    return pandas.DataFrame([transformed_inputs])


def op_transformation(output):
    op_tfm  = dict((("0", "unacceptable"), ("1", "acceptable"), 
                    ("2", "good deal"), ("3", "very good deal")))
    return op_tfm.get(str(output), output)


ip_transform = ip_transformation 
op_transform = op_transformation
#print(ip_transform(['vhigh', 'high', 2, 3, 'low', 'small', 'med']))
#print(op_transform("0"))