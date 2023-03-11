#include <queue>
#include <limits>
#include <cmath>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <experimental_heuristics.h>
#include<map>



const float INF = std::numeric_limits<float>::infinity();

// represents a single pixel
class Node {
  public:
    int pidx; // previous index
    int idx; // index in the flattened grid
    float cost; // cost of traversing this pixel
    int path_length; // the length of the path to reach this node

    Node(int i, float c, int path_length) : idx(i), cost(c), path_length(path_length) {}
};

// the top of the priority queue is the greatest element by default,
// but we want the smallest, so flip the sign
bool operator<(const Node &n1, const Node &n2) {
  return n1.cost > n2.cost;
}

// See for various grid heuristics:
// http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#S7
// L_\inf norm (diagonal distance)
inline float linf_norm(int i0, int j0, int i1, int j1) {
  return std::max(std::abs(i0 - i1), std::abs(j0 - j1));
}

// L_1 norm (manhattan distance)
inline float l1_norm(int i0, int j0, int i1, int j1) {
  return std::abs(i0 - i1) + std::abs(j0 - j1);
}




static PyObject *mastar(PyObject *self, PyObject *args) {
  const PyArrayObject* weights_object;
  const PyArrayObject* edges_object;
  const PyArrayObject* values_object;
  int h;
  int w;
  int m;
  const PyArrayObject* starts_obj;
  const PyArrayObject* goals_obj;
  int start_num;
  int goal_num;
  int diag_ok;
  int heuristic_override;

  if (!PyArg_ParseTuple(
        args, "OOOiiiOOiiii", // i = int, O = object
        &weights_object,
        &edges_object,
        &values_object,
        &h, &w, &m,
        &starts_obj, &goals_obj, &start_num, &goal_num,
        &diag_ok, &heuristic_override
        ))
    return NULL;

  int* weights = (int*) weights_object->data;
  int* edges = (int*) edges_object->data;
  float* values = (float*) values_object->data;
  int* starts = (int*) starts_obj->data;
  int* goals = (int*) goals_obj->data;

  std::map<std::pair<int, int>, float> value_map;

  for (int i=0; i<m; i++)
  {
      value_map.insert({{edges[i*2], edges[i*2+1]}, values[i]});
  }

  int* paths = new int[h * w];
  int path_length = -1;

  float* costs = new float[h * w];
  for (int i = 0; i < h * w; ++i)
    costs[i] = INF;

  std::priority_queue<Node> nodes_to_visit;
  

  for (int i; i<start_num; i++)
  {
    int start = starts[i];
    Node start_node(start, 0., 1);
    costs[start] = 0.;
    nodes_to_visit.push(start_node);
  }

  

  

  int prev_path = -1;

  int* nbrs = new int[8];
  int reached_goal = goals[0];
  //int goal_i = goal / w;
  //int goal_j = goal % w;
  //int start_i = start / w;
  //int start_j = start % w;

  heuristic_ptr heuristic_func = select_heuristic(heuristic_override);

  while (!nodes_to_visit.empty()) {
    // .top() doesn't actually remove the node
    Node cur = nodes_to_visit.top();
    bool bigbreak = false;
    for (int i=0; i < goal_num; i++){
        if (cur.idx == goals[i]) {
          path_length = cur.path_length;
          bigbreak = true;
          reached_goal = goals[i];
          break;
        }
    }
    if (bigbreak){
        break;
    }
    

    nodes_to_visit.pop();

    int row = cur.idx / w;
    int col = cur.idx % w;
    // check bounds and find up to eight neighbors: top to bottom, left to right
    nbrs[0] = (diag_ok && row > 0 && col > 0)          ? cur.idx - w - 1   : -1;
    nbrs[1] = (row > 0)                                ? cur.idx - w       : -1;
    nbrs[2] = (diag_ok && row > 0 && col + 1 < w)      ? cur.idx - w + 1   : -1;
    nbrs[3] = (col > 0)                                ? cur.idx - 1       : -1;
    nbrs[4] = (col + 1 < w)                            ? cur.idx + 1       : -1;
    nbrs[5] = (diag_ok && row + 1 < h && col > 0)      ? cur.idx + w - 1   : -1;
    nbrs[6] = (row + 1 < h)                            ? cur.idx + w       : -1;
    nbrs[7] = (diag_ok && row + 1 < h && col + 1 < w ) ? cur.idx + w + 1   : -1;

    int curLabel = weights[cur.idx];
    for (int i = 0; i < 8; ++i) {
      float heuristic_cost = INF;
      if (nbrs[i] >= 0) {
        // the sum of the cost so far and the cost of this move
        float new_cost = INF;
        float nc = 0;
        int neiLabel = weights[nbrs[i]];
        if (neiLabel == 0){
            new_cost = INF;
        }
        else if (neiLabel != curLabel){
            //if (value_map.find({curLabel-1, neiLabel-1}) == value_map.end()){
            //    new_cost = costs[cur.idx] + 1;
            //}
            //else{
            new_cost = costs[cur.idx] + value_map[{curLabel-1, neiLabel-1}];
            nc = value_map[{curLabel, neiLabel}];
            //}
        }
        else
        {
            new_cost = costs[cur.idx] + 1;
        }
        if (new_cost < costs[nbrs[i]]) {
          // estimate the cost to the goal based on legal moves
          // Get the heuristic method to use
          if (heuristic_override == DEFAULT) {
            if (diag_ok) {
              for (int j=0; j<goal_num; j++){
                  float heuristic_ = linf_norm(nbrs[i] / w, nbrs[i] % w, goals[j] / w, goals[j] % w);
                  if (heuristic_ < heuristic_cost){
                      heuristic_cost = heuristic_;
                  }
              }

            } else {
              
              for (int j=0; j<goal_num; j++){
                  float heuristic_ = l1_norm(nbrs[i] / w, nbrs[i] % w, goals[j] / w, goals[j] % w);
                  if (heuristic_ < heuristic_cost){
                      heuristic_cost = heuristic_;
                  }
              }
              
            }
          }

          // paths with lower expected cost are explored first
          float priority = new_cost + heuristic_cost;
          nodes_to_visit.push(Node(nbrs[i], priority, cur.path_length + 1));

          costs[nbrs[i]] = new_cost;
          paths[nbrs[i]] = cur.idx;
        }
      }
    }
  }

  PyObject *return_val;
  if (path_length >= 0) {
    npy_intp dims[2] = {path_length, 2};
    PyArrayObject* path = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT32);
    npy_int32 *iptr, *jptr;
    int idx = reached_goal;
    for (npy_intp i = dims[0] - 1; i >= 0; --i) {
        iptr = (npy_int32*) (path->data + i * path->strides[0]);
        jptr = (npy_int32*) (path->data + i * path->strides[0] + path->strides[1]);

        *iptr = idx / w;
        *jptr = idx % w;

        idx = paths[idx];
    }

    return_val = PyArray_Return(path);
  }
  else {
    return_val = Py_BuildValue(""); // no soln --> return None
  }

  delete[] costs;
  delete[] nbrs;
  delete[] paths;

  return return_val;
}





static PyMethodDef mastar_methods[] = {
    {"mastar", (PyCFunction)mastar, METH_VARARGS, "mastar"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mastar_module = {
    PyModuleDef_HEAD_INIT,"mastar", NULL, -1, mastar_methods
};

PyMODINIT_FUNC PyInit_astar(void) {
  import_array();
  return PyModule_Create(&mastar_module);
}
