def find_end_of_subtree(sentence, pos):
  if sentence[pos] != "(":
    return pos
  sum = 1
  while sum > 0:
    pos += 1
    if (pos >= len(sentence)):
      return -1
    if sentence[pos] == "(":
      sum += 1
    elif sentence[pos] == ")":
      sum -= 1
    if sum == 0:
      return pos
  return -1

def is_parsable2(sentence):
  ret = is_parsable(sentence)
  print(sentence, ret)
  return ret

def is_parsable3(sentence):
  if sentence[0] != "(":
    return False
  if find_end_of_subtree(sentence, 0) != len(sentence) - 1:
    return False
  if len(sentence) == 3:
    return True

  pos = find_end_of_subtree(sentence, 2) + 1
  if not is_parsable2(sentence[2:pos]):
    return False

  if sentence[1] == 'c' or sentence[1] == 'v' or sentence[1] == 'cart':
    return pos + 2 == len(sentence) 
  else:
    return is_parsable2(sentence[pos:(len(sentence) - 1)])

def is_parsable(sentence):
  if len(sentence) == 1:
    return True

  if sentence[0] != '(':
    return False
  if find_end_of_subtree(sentence, 0) + 1 != len(sentence):
    return False
  if len(sentence) == 3:
    return True

  pos = 2
  while pos + 1 < len(sentence):
    npos = find_end_of_subtree(sentence, pos) + 1
    if npos == 0:
      return False
    if not is_parsable(sentence[pos:npos]):
      return False
    pos = npos

  return True

def split_into_subtrees(sentence, max_size):
  if (len(sentence) <= 1):
    return []

  if len(sentence) <= max_size:
    return [sentence]

  ret = []
  pos = 2
  while pos + 1 < len(sentence):
    npos = find_end_of_subtree(sentence, pos) + 1
    ret.extend(split_into_subtrees(sentence[pos:npos], max_size))
    pos = npos

  return ret

def get_small_subtrees(sentence, min_size, max_size):
  if len(sentence) < min_size:
    return []

  ret = []
  if (len(sentence) <= max_size):
    ret.append(sentence)

  pos = 2
  while pos + 1 < len(sentence):
    npos = find_end_of_subtree(sentence, pos) + 1
    ret.extend(get_small_subtrees(sentence[pos:npos], min_size, max_size))
    pos = npos

  return ret

def find_subtree(tree, subtree):
  for i in range (len(tree)):
    for j in range (len(subtree)):
      if (tree[i + j] != subtree[j]):
        break

      if (j == len(subtree ) - 1):
        return i
