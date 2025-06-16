from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from copy import deepcopy
from qiskit.circuit import Instruction, QuantumRegister

def fault_simulation(fault_model, qc, shots=1024):
    if fault_model is None:
        qc_copy = deepcopy(qc)
        if not qc_copy.cregs or qc_copy.num_clbits == 0:
            qc_copy.measure_all()
        simulator = AerSimulator()
        result = simulator.run(qc_copy, shots=shots).result()
        return result.get_counts()

    faulty_qc = deepcopy(qc)
    target_qubits = fault_model.getQubits()  # e.g. [0, 1] for CX

    original_gate = fault_model.createOriginalGate()
    original_gate_name = original_gate.name

    new_data = []

    for instr in faulty_qc.data:
        op = instr.operation
        qargs = instr.qubits
        cargs = instr.clbits

        # Qubit *indices* of this instruction
        instr_qubit_indices = [faulty_qc.find_bit(q).index for q in qargs]

        # Only match gate if name AND qubit indices (and order!) match exactly
        if op.name == original_gate_name and instr_qubit_indices == target_qubits:
            # Generate faulty gate
            faulty_gate = fault_model.createFaultyGate(op)

            if isinstance(faulty_gate, Instruction):
                new_data.append((faulty_gate, qargs, cargs))
            elif isinstance(faulty_gate, QuantumCircuit):
                # Map faulty circuit's qubits to real circuit qubits
                for new_instr in faulty_gate.data:
                    new_op = new_instr.operation
                    new_qargs = [qargs[i] for i in range(len(new_instr.qubits))]
                    new_data.append((new_op, new_qargs, cargs))
            else:
                raise ValueError("Faulty gate must be Instruction or QuantumCircuit")
        else:
            new_data.append((op, qargs, cargs))

    faulty_qc.data = new_data

    if not faulty_qc.cregs or faulty_qc.num_clbits == 0:
        faulty_qc.measure_all()

    simulator = AerSimulator()
    result = simulator.run(faulty_qc, shots=shots).result()
    return result.get_counts()