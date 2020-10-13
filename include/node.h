#include <vector>
#include "tensor.h"

class Node
{
  public:
    std::vector<Node *> parents;
    std::vector<Node *> children;
    Tensor value;
    Tensor grad;
    bool waiting_for_backward = true;
    bool waiting_for_forward = true;
};
