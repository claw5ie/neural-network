#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdarg>
#include <cstdint>
#include <cassert>

template<class Type>
Type *malloc_or_exit(size_t count)
{
  Type *const data = (Type *)std::malloc(count * sizeof (Type));

  if (data == NULL)
  {
    std::fprintf(stderr,
                 "ERROR: failed to allocate %zu elements of "
                   "size %zu.\n",
                 count,
                 sizeof (Type));
    std::exit(EXIT_FAILURE);
  }

  return data;
}

struct Chunks
{
  char *data;
  char **offsets;
  size_t count;
};

Chunks allocate_aligned_chunks(size_t count, ...)
{
  auto const snap =
    [](size_t size) -> size_t
    {
      size += sizeof (void *) - 1;

      return size - size % sizeof (void *);
    };

  char **const offsets = malloc_or_exit<char *>(count + 1);

  va_list args;
  va_start(args, count);

  offsets[0] = (char *)0;
  for (size_t i = 1; i <= count; i++)
    offsets[i] = offsets[i - 1] + snap(va_arg(args, size_t));

  va_end(args);

  char *const data = malloc_or_exit<char>((uintptr_t)offsets[count]);

  for (size_t i = 0; i <= count; i++)
    offsets[i] = data + (uintptr_t)offsets[i];

  return { data, offsets, count + 1 };
}

double sigmoid(double x)
{
  return 1.0 / (std::exp(-x) + 1);
}

double dsigmoid(double y)
{
  return y * (1 - y);
}

struct Mat
{
  double *data;
  size_t rows;
  size_t cols;
};

double &get(const Mat &matrix, size_t row, size_t column)
{
  assert(row < matrix.rows && column < matrix.cols);

  return matrix.data[row * matrix.cols + column];
}

struct Vector
{
  double *data;
  size_t count;
};

double &get(const Vector &vector, size_t index)
{
  assert(index < vector.count);

  return vector.data[index];
}

double cost(const Vector &actual, const Vector &expected)
{
  assert(actual.count == expected.count);

  double cost_value = 0;

  for (size_t i = 0; i < actual.count; i++)
  {
    double const diff = get(actual, i) - get(expected, i);
    cost_value += diff * diff;
  }

  return cost_value / actual.count;
}

void compute_activations(
  const Vector &dest,
  const Mat &matrix,
  const Vector &vector,
  double (*activation)(double)
  )
{
  assert(matrix.cols == vector.count && dest.count == matrix.rows);

  for (size_t i = 0; i < matrix.rows; i++)
  {
    get(dest, i) = 0;

    for (size_t j = 0; j < matrix.cols; j++)
      get(dest, i) += get(matrix, i, j) * get(vector, j);

    get(dest, i) = activation(get(dest, i));
  }
}

struct Dataset
{
  double *data;
  size_t samples;
  size_t inputs;
  size_t outputs;
};

Vector input(const Dataset &dataset, size_t row)
{
  assert(row < dataset.samples);

  return { dataset.data + row * (dataset.inputs + dataset.outputs),
           dataset.inputs };
}

Vector output(const Dataset &dataset, size_t row)
{
  assert(row < dataset.samples);

  return { dataset.data +
             row * (dataset.inputs + dataset.outputs) +
             dataset.inputs,
           dataset.outputs };
}

Dataset read_dataset(const char *filepath)
{
  FILE *const file = std::fopen(filepath, "r");

  if (file == NULL)
  {
    std::fprintf(stderr,
                 "ERROR: failed to open file %s.\n",
                 filepath);
    std::exit(EXIT_FAILURE);
  }

  Dataset dataset;

  if (std::fscanf(file,
                  "%zu %zu %zu",
                  &dataset.samples,
                  &dataset.inputs,
                  &dataset.outputs) != 3)
  {
    std::fputs("ERROR: failed to read header of the file.\n",
               stderr);
    std::exit(EXIT_FAILURE);
  }

  size_t const doubles_to_read =
    dataset.samples * (dataset.inputs + dataset.outputs);

  dataset.data = malloc_or_exit<double>(doubles_to_read);

  for (size_t i = 0; i < doubles_to_read; i++)
  {
    if (std::fscanf(file, "%lf", &dataset.data[i]) != 1)
    {
      std::fprintf(stderr,
                   "ERROR: failed to read %zu values.\n",
                   doubles_to_read - i);
      std::exit(EXIT_FAILURE);
    }
  }

  std::fclose(file);

  return dataset;
}

struct NeuralNetwork
{
  Mat *weights;
  Vector *actvs;
  size_t *neuron_counts;
  size_t layers_count;
  char *raw_data;
};

double rand(double min, double max)
{
  return (double)std::rand() / RAND_MAX * (max - min) + min;
}

const Vector &input(const NeuralNetwork &net)
{
  return net.actvs[0];
}

NeuralNetwork create_neural_network(
  const size_t *neuron_counts,
  size_t layers
  )
{
  if (layers <= 1)
  {
    std::fputs("ERROR: number of layers should be more than 1.\n",
               stderr);
    std::exit(EXIT_FAILURE);
  }

  assert(neuron_counts[0] > 0);

  size_t network_size = neuron_counts[0] * sizeof (double);

  for (size_t i = 1; i < layers; i++)
  {
    assert(neuron_counts[i] > 0);

    network_size += (neuron_counts[i - 1] * neuron_counts[i] + neuron_counts[i]) * sizeof (double);
  }

  NeuralNetwork net;

  Chunks const chunks = allocate_aligned_chunks(
    4,
    (layers - 1) * sizeof (Mat),
    layers * sizeof (Vector),
    layers * sizeof (size_t),
    network_size
    );

  net.weights = (Mat *)chunks.offsets[0];
  net.actvs = (Vector *)chunks.offsets[1];
  net.neuron_counts = (size_t *)chunks.offsets[2];
  net.layers_count = layers;
  net.raw_data = chunks.offsets[0];

  for (double *i = (double *)chunks.offsets[3];
       i < (double *)chunks.offsets[4];
       i++)
  {
    *i = rand(-1, 1);
  }

  double *next = (double *)chunks.offsets[3];

  net.actvs[0].count = neuron_counts[0];
  net.actvs[0].data = next;

  next += net.actvs[0].count;

  for (size_t i = 1; i < layers; i++)
  {
    auto &weights = net.weights[i - 1];

    weights.rows = neuron_counts[i];
    weights.cols = neuron_counts[i - 1];
    weights.data = next;

    next += weights.rows * weights.cols;

    auto &actv = net.actvs[i];

    actv.count = neuron_counts[i];
    actv.data = next;

    next += actv.count;
  }

  std::memcpy(net.neuron_counts,
              neuron_counts,
              layers * sizeof (size_t));

  assert(next == (double *)chunks.offsets[4]);

  std::free(chunks.offsets);

  return net;
}

void feed(const NeuralNetwork &net, const Vector &input)
{
  assert(net.neuron_counts[0] == input.count);

  std::memcpy(net.actvs[0].data,
              input.data,
              input.count * sizeof (double));

  for (size_t i = 1; i < net.layers_count; i++)
  {
    compute_activations(
      net.actvs[i], net.weights[i - 1], net.actvs[i - 1], sigmoid
      );
  }
}

const Vector &output(const NeuralNetwork &net)
{
  return net.actvs[net.layers_count - 1];
}

void train(
  const NeuralNetwork &net,
  const Dataset &dataset,
  double step,
  double eps
  )
{
  assert(input(net).count == dataset.inputs &&
         output(net).count == dataset.outputs);

  size_t max_matrix_count = 0;

  for (size_t i = 0; i < net.layers_count - 1; i++)
  {
    max_matrix_count = std::max(
      max_matrix_count, net.weights[i].rows * net.weights[i].cols
      );
  }

  double *const buffer =
    malloc_or_exit<double>(2 * max_matrix_count);

  Mat front = { buffer, 0, 0 };
  Mat back = { buffer + max_matrix_count, 0, 0 };

  auto const adjust_weights =
    [&net, &front, &back, step](const Vector &expected)
    {
      {
        const Vector &actual = output(net);
        size_t const last_layer = net.layers_count - 1;

        front.rows = net.neuron_counts[last_layer];
        front.cols = net.neuron_counts[last_layer - 1];

        for (size_t row = 0;
             row < net.weights[last_layer - 1].rows;
             row++)
        {
          double const delta = (get(actual, row) - get(expected, row)) * dsigmoid(get(actual, row));

          for (size_t col = 0;
               col < net.weights[last_layer - 1].cols;
               col++)
          {
            double &weight = get(net.weights[last_layer - 1], row, col);

            get(front, row, col) = weight * delta;
            weight -= step * delta * get(net.actvs[last_layer - 1], col);
          }
        }
      }

      std::swap(front, back);

      for (size_t layer = net.layers_count - 1; layer-- > 1; )
      {
        front.rows = net.neuron_counts[layer];
        front.cols = net.neuron_counts[layer - 1];

        for (size_t row = 0;
             row < net.weights[layer - 1].rows;
             row++)
        {
          double delta = 0;
          for (size_t k = 0; k < net.neuron_counts[layer + 1]; k++)
            delta += get(back, k, row);

          delta *= dsigmoid(get(net.actvs[layer], row));

          for (size_t col = 0;
               col < net.weights[layer - 1].cols;
               col++)
          {
            double &weight = get(net.weights[layer - 1], row, col);

            get(front, row, col) = weight * delta;
            weight -= step * delta * get(net.actvs[layer - 1], col);
          }
        }

        std::swap(front, back);
      }
    };

  double error;

  do
  {
    error = 0;
    for (size_t i = 0; i < dataset.samples; i++)
    {
      feed(net, input(dataset, i));
      error += cost(output(net), output(dataset, i));
      adjust_weights(output(dataset, i));
    }

    error /= dataset.samples;
  } while (error > eps);

  std::free(buffer);
}

int main(int argc, char **argv)
{
  assert(argc >= 3);

  std::srand(10234);

  Dataset dataset = read_dataset(argv[1]);

  size_t const counts[4] = { dataset.inputs, 4, 3, dataset.outputs };
  NeuralNetwork net = create_neural_network(counts, 4);

  train(net, dataset, 0.05, 0.00001);

  std::free(dataset.data);

  dataset = read_dataset(argv[2]);

  {
    double average_cost = 0;

    for (size_t i = 0; i < dataset.samples; i++)
    {
      feed(net, input(dataset, i));
      average_cost += cost(output(net), output(dataset, i));
    }

    std::printf("avarage cost: %lg.\n",
                average_cost / dataset.samples);
  }

  std::free(dataset.data);
  std::free(net.raw_data);
}
