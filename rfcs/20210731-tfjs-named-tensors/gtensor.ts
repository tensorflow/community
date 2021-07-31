/* Graphical Tensors: each dimension is named.

Related stuff:

* PyTorch Named Tensors:
https://pytorch.org/docs/stable/named_tensor.html
https://github.com/harvardnlp/namedtensor/
http://nlp.seas.harvard.edu/NamedTensor aka Tensors Considered Harmful.

* XArray: pandas inspired numerical library for python.
http://xarray.pydata.org/en/stable/

* Haskell:
https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html

* Random github bug discussion (a bit confused/confusing):
https://github.com/KhronosGroup/NNEF-Tools/issues/3

TODO: Cleanup the typing using this library: https://github.com/unional/type-plus
  Looks likely that we can get this to check sizes too, although it may make types
  painful/annoying to read.

*/
import * as tf from '@tensorflow/tfjs';
import * as tf_init from '@tensorflow/tfjs-layers/dist/initializers';

type DName = string|number|symbol;

export interface DimensionData<G extends DName, D extends G> {
  name: D;
  size: number;
  gtensor: GTensor<G>;
  index: number;
}

// There's one lucky case when no permutations are needed... that's when the last dimension of the
// first tensor matches (has the same size) as the second-to-last dimension of the second tensor.
function luckyCaseMultiply<D extends G1 & G2, G1 extends DName, G2 extends DName>(
  g1: GTensor<G1>, g2: GTensor<G2>): GTensor<Exclude<G1|G2, D>> {
  const resultTensor = tf.matMul(g1.tensor, g2.tensor);
  const newNames =
    (g2.dimNames.slice(0, g1.dimNames.length - 3) as DName[])
    .concat(g1.dimNames.slice(0, g1.dimNames.length - 2))
    .concat([g2.dimNames[g2.dimNames.length - 1]]);
  return new GTensor(resultTensor, newNames as (Exclude<G1|G2, D>)[]);
}

interface DotError_MustPassDimensionToDot {
  _DotError_MustPassDimensionToDot: ['DotError_MustPassDimensionToDot']
}

interface DotError_DimensionNamesMustBeEqual<D1, D2> {
  _DotError_DimensionNamesMustBeEqual: ['DotError_DimensionNamesMustBeEqual', D1, D2]
}

interface DotError_NotAllowedNameOutsideDot<OtherNames> {
  _DotError_NotAllowedNameOutsideDot: ['DotError_NotAllowedNameOutsideDot', OtherNames]
}

type DotCompatibleDimension<M1 extends DName, D1 extends M1, M2 extends DName, D2 extends M2> =
  D2 extends never ? DotError_MustPassDimensionToDot :
  D1 extends D2 ? D2 extends D1
  ? Exclude<M1 & M2, D1 & D2> extends never
    ? Dimension<M2, D2>
    : DotError_NotAllowedNameOutsideDot<Exclude<M1 & M2, D2>>
    : DotError_DimensionNamesMustBeEqual<D2,D1>
  : DotError_DimensionNamesMustBeEqual<D1,D2>;

// type OnlyDsInterection<D1 extends G1, D2 extends G2, G1 extends string, G2 extends string> =
//   // D1 = D2, lets call this D
//   D2 extends D1 ? D1 extends D2 ?
//   // G intersects G2 only at D
//   (Exclude<G1 & G2, D2> extends never ? Dimension<G2, D2> : ErrorGTensorsOtherIntersectionNames<Exclude<G1 & G2, D2>>)
//   : ErrorFirstDimFailsToExtendSecond<D1,D2> : ErrorFirstDimFailsToExtendSecond<D2,D1>;

export function dot<D1 extends G1, D2 extends G2, G1 extends DName, G2 extends DName>(
  d1: Dimension<G1, D1>,
  maybed2: DotCompatibleDimension<G1,D1,G2,D2>
): GTensor<Exclude<G1|G2, D1>> {
  // TODO: maybe we canmake the type system do more for us...
  //   D extends D2 ? (D2 extends D ? GTensor<Exclude<G1|G2, D>> : never) : never
  // TODO(ldixon): this is a cheep way to find a local-optima in terms of tensor permutations. The
  // 'right' way to do this is of course to track the graph and do a global
  // compilation/optimization. But this will be reasonable for now I expect (and probably still
  // better than what I'd be able to do eaily in my head.
  //
  // TODO: think about if the 'never' below is needed.
  let d2 = maybed2 as never as Dimension<G2, D2>;
  if (d1.isLastDim) {
    if (d2.isSecondLastDim) {
      return luckyCaseMultiply(d1.gtensor, d2.gtensor) as never as GTensor<Exclude<G1|G2, D1>>;
    } else if (d2.isSecondDim) {
      luckyCaseMultiply(d1.gtensor, d2.gtensor.transpose());
    } else {
      luckyCaseMultiply(d1.gtensor, d2.gtensor._permuteIndexToSecondLast(d2.index));
    }
  } else if (d1.isSecondLastDim) {
    if (d2.isLastDim) {
      return luckyCaseMultiply(d2.gtensor, d1.gtensor) as never as GTensor<Exclude<G1|G2, D1>>;
    } else if (d2.isFirstDim) {
      return luckyCaseMultiply(d2.gtensor.transpose(), d1.gtensor) as never as GTensor<Exclude<G1|G2, D1>>;
    } else {
      return luckyCaseMultiply(d2.gtensor._permuteIndexToLast(d2.index), d1.gtensor) as never as GTensor<Exclude<G1|G2, D1>>;;
    }
  }

  // General case... have to permute both tensors to do the multiplication. AWESOME: type-checking
  // index names fixed a bug here, I had accidentally multiplied the matrix with itself.
  return luckyCaseMultiply(
    d1.gtensor._permuteIndexToLast(d1.index),
    d2.gtensor._permuteIndexToSecondLast(d2.index)) as never as GTensor<Exclude<G1|G2, D1>>;
}


interface LiftError_DimInInput<D> {
  _LiftError_DimInInput: ['LiftError_DimInInput', D]
}

interface LiftError_DimInOutput<D> {
  _LiftError_DimMustBeInFnOutput: ['LiftError_DimInOutput', D]
}


type DimensionFnToLift<D extends DName, G extends DName, G2 extends DName> =
  D extends G  ? LiftError_DimInInput<D> :
  D extends G2 ? LiftError_DimInOutput<D> : D

export function liftFnOverDim<D extends DName, G extends DName, G2 extends DName>(
  liftDim: DimensionFnToLift<D,G,G2>,
  toLiftFn: (input: Dims<G>) => Dims<G2>): (input: Dims<G|D>) => Dims<G2|D> {
  function liftedFn(input: Dims<G|D>): Dims<G2|D> {
    if ((liftDim as DName) in input) {
      throw new ValueError(`The lift dimension ${liftDim} already occurs in input's dimensions: ${Object.keys(input)}`);
    }
    const unstacked_dims = input[liftDim as D].unstack() as never as Dims<G>[];
    return stack(liftDim as D, unstacked_dims.map(toLiftFn));
  }
  return liftedFn;
}

export function liftMapFnOverDim<
    D extends DName, // The new dimension being lifted over.
    G extends DName, // The dimensions of the input.
    // A mapping from the name of each output of toLiftFn to the dimensions of that output.
    MapDim extends { [key in keyof MapDim]: MapDim[keyof MapDim] },
>(
  liftDim: DimensionFnToLift<D,G,MapDim[keyof MapDim]>,
  toLiftFn: (input: Dims<G>) => { [key in keyof MapDim]: Dims<MapDim[key]> },
): (input: Dims<G|D>) => { [key in keyof MapDim]: Dims<MapDim[key]|D> } {

  function liftedFn(input: Dims<G|D>): { [key in keyof MapDim]: Dims<MapDim[key]|D> } {
    if ((liftDim as string) in input) {
      throw new ValueError(`The lift dimension ${liftDim} already occurs in input's dimensions: ${Object.keys(input)}`);
    }
    const unstacked_dims = input[liftDim as D].unstack() as never as Dims<G>[];
    const unstackedApplications = unstacked_dims.map(toLiftFn);
    const stackedApplications = {} as { [key in keyof MapDim]: Dims<MapDim[key]|D> };
    for(const key of Object.keys(unstackedApplications[0]) as (keyof MapDim)[]) {
      const toStack = unstackedApplications.map(a => a[key] as Dims<MapDim[keyof MapDim]>);
      stackedApplications[key] = stack(liftDim as D, toStack);
    }
    return stackedApplications;
  }
  return liftedFn;
}


// G is the set of all names in the tensor. D is the specific name of this dimension.
export class Dimension<G extends DName, D extends G> implements DimensionData<G, D> {
  name: D;
  size: number;
  gtensor: GTensor<G>;
  index: number;

  constructor(e: DimensionData<G, D>) {
    this.name = e.name;
    this.size = e.size;
    this.gtensor = e.gtensor;
    this.index = e.index;
  }

  get dtype(): tf.DataType {
    return this.gtensor.tensor.dtype;
  }

  get isFirstDim(): boolean {
    return (this.index === 0);
  }
  get isSecondDim(): boolean {
    return (this.index === 1);
  }
  get isLastDim(): boolean {
    return (this.index === this.gtensor.tensor.shape.length);
  }
  get isSecondLastDim(): boolean {
    return (this.index === this.gtensor.tensor.shape.length);
  }

  _dot<D2 extends G2, G2 extends DName>(
    d2: DotCompatibleDimension<G,D,G2,D2>
  ): GTensor<Exclude<G | G2, D>> {
    return dot(this, d2);
  }

  dot<D2 extends G2, G2 extends DName>(
    d2: DotCompatibleDimension<G,D,G2,D2>
  ): Dims<Exclude<G | G2, D>> {
    return this._dot(d2).dim;
  }

  // softmax<D2 extends G2, G2 extends DName>(
  //   d2: DotCompatibleDimension<G,D,G2,D2>
  // ): Dims<Exclude<G | G2, D>> {
  //   tf.softmax(this.gtensor.tensor)
  // }

  _rename<T extends DName>(newName: T): GTensor<Exclude<G, D> | T> {
    // TODO: shouldn't TS be able to infer that typeod(this.name) extends G? It's specified in the
    // contrains for the class...?
    return this.gtensor.rename(this.name as never, newName) as GTensor<Exclude<G, D> | T>;
  }

  rename<T extends DName>(newName: T): Dims<Exclude<G, D> | T> {
    return this._rename(newName).dim;
  }

  _unstack(): GTensor<Exclude<G, D>>[] {
    const tensors = tf.unstack(this.gtensor.tensor, this.index);
    const newDimNames = ([...this.gtensor.dimNames].splice(this.index, 1) as Exclude<G, D>[]);
    return tensors.map(t => new GTensor<Exclude<G, D>>(t, newDimNames));
  }

  unstack(): Dims<Exclude<G, D>>[] {
    return this._unstack().map(g => g.dim);
  }
  // pairwise_add(d2: Dimension): GTensor;
  // pairwise_mult(d2: Dimension): GTensor;
}

export class ValueError extends Error {}


function gtensorOfDims<G extends DName>(dims: Dims<G>): GTensor<G> {
  // Technically, we don't know the dimension is G... but it doesn't matter, this makes TS happy.
  // In theory I think `unknown` should replace the second G.
  const d = Object.values(dims)[0] as Dimension<G, G>;
  if (!d) {
    throw new ValueError('gtensorOfDims: empty set of dimensions');
  }

  return d.gtensor;
}

//
export function stackGtensors<G extends DName, NewD extends DName>(
  newDimName: NewD,
  gtensors: GTensor<G>[],
): GTensor<G|NewD> {
  if(gtensors.length === 0) {
    throw new ValueError('stackDims was empty');
  }
  const tensors = gtensors.map(g => g.tensor);
  const newTensor = tf.stack(tensors);
  const newDimNames = [newDimName, ...gtensors[0].dimNames]
  return new GTensor(newTensor, newDimNames);
}
export function stack<G extends DName, NewD extends DName>(
  newDimName: NewD,
  stackDims: Dims<G>[],
): Dims<G|NewD> {
  const gtensors = stackDims.map(gtensorOfDims);
  return stackGtensors(newDimName, gtensors).dim;
}

export type Dims<G extends DName> = { [key in G]: Dimension<G, key> };

export class GTensor<G extends DName> {
  // TODO: the type-system fails here because we can't force dim to always have all the keys of T,
  // and for the key-name to match the Dimension<T>.
  //
  // The dimensions in the GTensor.
  dim!: Dims<G>;
  tensor: tf.Tensor;
  dimNames: G[];

  constructor(tensor: tf.Tensor, dimNames: G[]) {
    if(tensor.shape.length !== dimNames.length) {
      throw new ValueError(`tensor.shape.length: ${tensor.shape.length} should be the same as dimNames.length: ${dimNames}.`);
    }
    this.tensor = tensor;
    this.dimNames = dimNames;

    this._resetDim();
  }

  gshape(): { [key in G]: number } {
    const gshape = {} as { [key in G]: number };
    for (let i = 0; i < this.dimNames.length; i++) {
      gshape[this.dimNames[i]] = this.tensor.shape[i];
    }
    return gshape;
  }

  _resetDim() {
    this.dim = {} as Dims<G>;
    for (let i = 0; i < this.dimNames.length; i++) {
      const dim_i = new Dimension({
        name: this.dimNames[i],
        index: i,
        size: this.tensor.shape[i],
        gtensor: this,
      });
      (this.dim as {[k:string]: Dimension<G, any>})[dim_i.name as string] = dim_i;
    }
  }

  public transpose(): GTensor<G> {
    return new GTensor<G>(tf.transpose(this.tensor), this.dimNames.slice().reverse());
  }

  // Rename a set of dimensions.
  public renaming<ReplacedNames extends G, NewNames extends DName>(
    renaming: { [Key in ReplacedNames]: NewNames }
    // from: { [fromKey in G extends T1 ? T1 : never]: 'from' },
    // to: { [toKey in T2]: 'to' },
  ): GTensor<Exclude<G, ReplacedNames>|NewNames> {

    const newDimNames = [...this.dimNames] as never as  NewNames[];

    for (const key in renaming) {
      const index = this.dim[key].index;
      newDimNames[index as any] = renaming[key];
    }

    return new GTensor<Exclude<G, ReplacedNames>|NewNames>(this.tensor, newDimNames);
  }

  // Rename a single dimension.
  public rename<T1 extends DName, T2 extends DName>(
    // { [key in G]: T2 }
    fromName: G extends T1 ? T1 : never,
    toName: T2
    // from: { [fromKey in G extends T1 ? T1 : never]: 'from' },
    // to: { [toKey in T2]: 'to' },
  ): GTensor<Exclude<G, T1>|T2> {

    // const fromName = Object.keys(from)[0] as string; // T1;
    // const toName = Object.keys(to)[0] as ``;

    const i = this.dimNames.findIndex(n => (n as DName) === fromName);
    if (i === undefined) {
        throw new ValueError(`${fromName} is missing from ${this.dimNames}`);
    }

    const newDimNames =
      [...this.dimNames].splice(i, 1, toName as never) as (Exclude<G, T1>|T2)[];
    return new GTensor<Exclude<G, T1>|T2>(this.tensor, newDimNames);
  }

  public _permuteIndexTo(i:number, new_i:number): GTensor<G> {
    // hack/trick to start with the identity permutation [0,1,...n].
    const permutation = this.dimNames.map((s, i) => i);
    // Now swap the last and the ith index.
    //
    // TODO(ldixon): I heard that some permutations are cheeper than others, so is there some
    // super-smart way to do an optimal permutation?
    const lastIndex = new_i;
    permutation[i] = lastIndex;
    permutation[lastIndex] = i;
    const oldLastName = this.dimNames[lastIndex];
    const newLastName = this.dimNames[i];
    const newDimNames = this.dimNames.slice();
    newDimNames[lastIndex] = newLastName;
    newDimNames[i] = oldLastName;
    return new GTensor<G>(tf.transpose(permutation), newDimNames);
  }

  _permuteIndexToLast(i:number): GTensor<G> {
    return this._permuteIndexTo(i, this.dimNames.length - 1);
  }

  _permuteIndexToSecondLast(i:number): GTensor<G> {
    return this._permuteIndexTo(i, this.dimNames.length - 2);
  }

  public permuteDimNameToLast(name: G): GTensor<G> {
    return this._permuteIndexToLast(this.dim[name].index);
  }
}

export interface InitializerConfig {
  // Only one of these should be specified.
  tuncNormal?: tf_init.TruncatedNormalArgs;
  zeros?: {};
  ones?: {};
  constant?: tf_init.ConstantArgs;
}

export function makeInitializer(config: InitializerConfig) {
  if (config.tuncNormal) {
    return tf.initializers.truncatedNormal(config.tuncNormal);
  } else if (config.zeros) {
    return tf.initializers.zeros();
  } else if (config.ones) {
    return tf.initializers.ones();
  } else if (config.constant) {
    return tf.initializers.constant(config.constant);
  }

  throw new ValueError('need to specify an initalizer config');
}

export function fromInitializer<T extends string>(
  dims: { [key in T]: number },
  initialiser:  tf_init.Initializer, dtype?: tf.DataType) {
  const dimNames = Object.values(dims) as T[];
  const shape = dimNames.map((n: T) => dims[n]);
  return new GTensor(initialiser.apply(shape, dtype), dimNames);
}

export function makeTruncNormal<T extends string>(dims: { [key in T]: number },
  truncNormalConfig?: tf_init.TruncatedNormalArgs, dtype?: tf.DataType) {
  return fromInitializer(
    dims, tf.initializers.truncatedNormal(truncNormalConfig || {}), dtype);
}

export function makeZeros<T extends string>(dims: { [key in T]: number }, dtype?: tf.DataType) {
  return fromInitializer(dims, tf.initializers.zeros(), dtype);
}

export function makeOnes<T extends string>(dims: { [key in T]: number }, dtype?: tf.DataType) {
  return fromInitializer(dims, tf.initializers.ones(), dtype);
}

export function makeConstant<T extends string>(dims: { [key in T]: number },
  constant: number, dtype?: tf.DataType) {
  return fromInitializer(dims, tf.initializers.constant({value: constant}), dtype);
}

