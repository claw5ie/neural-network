#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>

size_t sum(const size_t *values, size_t count)
{
  size_t sum = 0;
  while (count-- > 0)
    sum += values[count];

  return sum;
}

double sigmoid(double x)
{
  return 1.0 / (exp(-x) + 1);
}

double dsigmoid(double y)
{
  return y * (1 - y);
}

struct Mat
{
  double *data;
  size_t rows,
    cols;

  double &operator()(size_t row, size_t column);

  double operator()(size_t row, size_t column) const;
};

double &Mat::operator()(size_t row, size_t column)
{
  assert(row < rows && column < cols);

  return data[row * cols + column];
}

double Mat::operator()(size_t row, size_t column) const
{
  assert(row < rows && column < cols);

  return data[row * cols + column];
}

struct Vector
{
  double *data;
  size_t count;

  double &operator[](size_t index);

  double operator[](size_t index) const;
};

double &Vector::operator[](size_t index)
{
  assert(index < count);

  return data[index];
}

double Vector::operator[](size_t index) const
{
  assert(index < count);

  return data[index];
}

double cost(const Vector &actual, const Vector &expected)
{
  assert(actual.count == expected.count);

  double cost_value = 0;

  for (size_t i = 0; i < actual.count; i++)
  {
    double const diff = actual[i] - expected[i];
    cost_value += diff * diff;
  }

  return cost_value / actual.count;
}

void multiply_in_place(
  Vector &dest, const Mat &matrix, const Vector &vector
  )
{
  assert(matrix.cols == vector.count && dest.count == matrix.rows);

  for (size_t i = 0; i < matrix.rows; i++)
  {
    dest[i] = 0;
    for (size_t j = 0; j < matrix.cols; j++)
      dest[i] += matrix(i, j) * vector[j];
  }
}

void map(double (*func)(double), Vector &dest)
{
  for (size_t i = 0; i < dest.count; i++)
    dest[i] = func(dest[i]);
}

struct MemoryPool
{
  char *data;
  size_t size;
  size_t reserved;

  static MemoryPool allocate(size_t bytes);

  void *reserve(size_t bytes);
};

MemoryPool MemoryPool::allocate(size_t bytes)
{
  return { new char[bytes], 0, bytes };
}

void *MemoryPool::reserve(size_t bytes)
{
  if (size + bytes > reserved)
  {
    std::cerr << "ERROR: failed to reserve "
              << bytes
              << " bytes: out of reserved memory.\n";
    std::exit(EXIT_FAILURE);
  }

  void *const begin = (void *)(data + size);

  size += bytes;

  return begin;
}

struct NeuralNetwork
{
  Mat *weights;
  Vector *actvs;
  size_t *neuron_counts;
  size_t layers_count;
  Mat front;
  Mat back;
  MemoryPool pool;

  static NeuralNetwork create(const size_t *neuron_counts, size_t layers);

  void feedforward(const Vector &input);

  void backpropagate(
    const Vector *input,
    const Vector *output,
    size_t count,
    double eps
    );

  void adjust_weights(const Vector &expected, double eps);

  const Vector &input() const;

  const Vector &output() const;
};

double rand_range(double min, double max)
{
  return (double)rand() / RAND_MAX * (max - min) + min;
}

NeuralNetwork NeuralNetwork::create(const size_t *neuron_counts, size_t layers)
{
  if (layers <= 1)
  {
    std::cerr << "ERROR: number of layers should be more than 1.\n";
    std::exit(EXIT_FAILURE);
  }

  assert(neuron_counts[0] > 0);

  size_t max_network_size = neuron_counts[0] * sizeof (double);
  size_t max_delta_matrix_size = 0;

  for (size_t i = 1; i < layers; i++)
  {
    assert(neuron_counts[i] > 0);

    size_t const matrix_size =
      neuron_counts[i - 1] * neuron_counts[i] * sizeof (double);

    max_network_size += matrix_size + neuron_counts[i] * sizeof (double);
    max_delta_matrix_size = std::max(max_delta_matrix_size, matrix_size);
  }

  NeuralNetwork net;
  size_t const sizes[6] = {
    (layers - 1) * sizeof (Mat),
    layers * sizeof (Vector),
    layers * sizeof (size_t),
    max_network_size,
    max_delta_matrix_size,
    max_delta_matrix_size
  };

  size_t const total_size = sum(sizes, sizeof (sizes) / sizeof (*sizes));

  net.pool = MemoryPool::allocate(total_size);
  net.weights = (Mat *)net.pool.reserve(sizes[0]);
  net.actvs = (Vector *)net.pool.reserve(sizes[1]);
  net.neuron_counts = (size_t *)net.pool.reserve(sizes[2]);
  net.layers_count = layers;

  for (double *i = (double *)(net.pool.data + net.pool.size),
         *const end = i + max_network_size / sizeof (double);
       i < end;
       i++)
  {
    *i = rand_range(-1, 1);
  }

  net.actvs[0].count = neuron_counts[0];
  net.actvs[0].data = (double *)net.pool.reserve(
    net.actvs[0].count * sizeof (double)
    );

  for (size_t i = 1; i < layers; i++)
  {
    auto &weights = net.weights[i - 1];

    weights.rows = neuron_counts[i];
    weights.cols = neuron_counts[i - 1];
    weights.data = (double *)net.pool.reserve(
      weights.rows * weights.cols * sizeof (double)
      );

    auto &actv = net.actvs[i];

    actv.count = neuron_counts[i];
    actv.data = (double *)net.pool.reserve(
      actv.count * sizeof (double)
      );
  }

  net.front.data = (double *)net.pool.reserve(max_delta_matrix_size);
  net.back.data = (double *)net.pool.reserve(max_delta_matrix_size);

  assert(net.pool.size == net.pool.reserved);

  std::memcpy(net.neuron_counts, neuron_counts, layers * sizeof (size_t));

  return net;
}

void NeuralNetwork::feedforward(const Vector &input)
{
  assert(neuron_counts[0] == input.count);

  std::memcpy(actvs[0].data, input.data, input.count * sizeof (double));

  for (size_t i = 1; i < layers_count; i++)
  {
    // Maybe should apply all of those operations at once.
    multiply_in_place(actvs[i], weights[i - 1], actvs[i - 1]);
    map(sigmoid, actvs[i]);
  }
}

const Vector &NeuralNetwork::output() const
{
  return actvs[layers_count - 1];
}

double derivative(
  NeuralNetwork &net,
  const Vector *input,
  const Vector *output,
  size_t layer,
  size_t row,
  size_t col
  )
{
  const double h = 0.00001;
  double &weight = net.weights[layer](row, col);
  weight -= h;
  net.feedforward(*input);
  double const cost_before = cost(net.output(), *output);
  /* double const cost_before = net->activations[2].data[1]; */
  weight += 2 * h;
  net.feedforward(*input);
  double const cost_after = cost(net.output(), *output);
  /* double const cost_after = net->activations[2].data[1]; */
  weight -= h;
  net.feedforward(*input);

  return (cost_after - cost_before) / (2 * h);
}

void NeuralNetwork::backpropagate(
  const Vector *input,
  const Vector *output,
  size_t count,
  double eps
  )
{
  double error;
  do
  {
    error = 0;
    for (size_t i = 0; i < count; i++)
    {
      feedforward(input[i]);
      error += cost(this->output(), output[i]);
      adjust_weights(output[i], eps);
    }

    error /= count;
  } while (error > eps);
}

void NeuralNetwork::adjust_weights(const Vector &expected, double eps)
{
  {
    const Vector &actual = this->output();
    size_t const last_layer = layers_count - 1;

    front.rows = neuron_counts[last_layer];
    front.cols = neuron_counts[last_layer - 1];

    for (size_t row = 0; row < weights[last_layer - 1].rows; row++)
    {
      double const delta =
        (actual[row] - expected[row]) * dsigmoid(actual[row]);

      for (size_t col = 0; col < weights[last_layer - 1].cols; col++)
      {
        double &weight = weights[last_layer - 1](row, col);

        front(row, col) = weight * delta;
        weight -= eps * delta * actvs[last_layer - 1][col];
        // std::cout << delta * actvs[last_layer - 1][col] -
        //   derivative(*this, &input(), &expected, last_layer - 1, row, col)
        //           << '\n';
      }
    }
  }

  std::swap(front, back);

  for (size_t layer = layers_count - 1; layer-- > 1; )
  {
    front.rows = neuron_counts[layer];
    front.cols = neuron_counts[layer - 1];

    for (size_t row = 0; row < weights[layer - 1].rows; row++)
    {
      double delta = 0;
      for (size_t k = 0; k < neuron_counts[layer + 1]; k++)
        delta += back(k, row);

      delta *= dsigmoid(actvs[layer][row]);

      for (size_t col = 0; col < weights[layer - 1].cols; col++)
      {
        double &weight = weights[layer - 1](row, col);

        front(row, col) = weight * delta;
        weight -= eps * delta * actvs[layer - 1][col];
        // std::cout << delta * actvs[layer - 1][col] -
        //   derivative(*this, &input(), expected, layer - 1, row, col) << '\n';
      }
    }

    std::swap(front, back);
  }
}

int main()
{
  srand(10234);

  size_t const counts[4] = { 3, 4, 3, 2 };
  NeuralNetwork net = NeuralNetwork::create(counts, 4);

  double in[] = { 1, 0, 1 };
  double tar[] = { 0.4, 0.8 };
  Vector input = { in, 3 };
  Vector target = { tar, 2 };

  net.backpropagate(&input, &target, 1, 0.0005);

  net.feedforward(input);
  const Vector &output = net.output();

  for (size_t i = 0; i < output.count; i++)
    std::cout << output[i] << (i + 1 < output.count ? ' ' : '\n');
}
