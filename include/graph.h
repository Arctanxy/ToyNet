#include <vector>

class Graph
{
  public:
    Graph(){};
    ~Graph(){};
    std::vector<Node*> nodes;
    int add_node(Node *node);
    int backward();
    int zero_grad();
};

int Graph::add_node(Node * node)
{
  if(node == NULL) return 1;
  this->nodes.emplace_back(node);
  return 0;
}

int zero_grad()
{
  for(Node* n:this->nodes)
  {
    n->zero_grad();
    n->waiting_for_backward = True;
    n->waiting_for_forward = True;
  }
  return 0;
}

int backward()
{
  for(Node *n:this->nodes)
  {
    n->backward();
  }
  return 0;
}
