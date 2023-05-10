import numpy as np


class BVH:
    def __init__(self):
        self.bvh_rot_map = {"Xrotation": "x", "Yrotation": "y", "Zrotation": "z"}
        self.bvh_map_num = {"x": 0, "y": 1, "z": 2}
        self.bvh_pos_map_num = {"Xposition": 0, "Yposition": 1, "Zposition": 2}
        self.inv_bvh_rot_map = {v: k for k, v in self.bvh_rot_map.items()}

    def load(self, filename):
        """
        Reads a BVH file and returns a dictionary with the data.

        Parameters
        ----------
        filename : str
            path to the file.

        Returns
        ------
        self.data : dict
            dictionary with the data.
        self.data["names"] : list[str]
            ith-element contain the name of ith-joint.
        self.data["offsets"] : np.array
            ith-element contain the offset of ith-joint wrt. its parent joint.
        self.data["parents"] : list[int]
            ith-element contain the parent of the ith joint.
        self.data["rot_order"] : np.array
            axis=0 - joints, axis=1 - order per channel
        self.data["positions"] : np.array
            axis=0 - frames, axis=1 - joints, axis=2 - local positions.
        self.data["rotations"] : np.array
            axis=0 - frames, axis=1 - joints, axis=2 - local rotations.
        self.data["frame_time"] : float
            time between two frames in seconds.
        """
        f = open(filename, "r")

        names = []
        offsets = []
        parents = []
        position_order = []
        rot_order = []
        channels = []

        current = None
        is_end_site = False
        reading_frames = False
        frame = 0

        for line in f:
            if not reading_frames:
                if "HIERARCHY" in line or "MOTION" in line or "{" in line:
                    continue

                if "ROOT" in line or "JOINT" in line:
                    names.append(line.split()[1])
                    offsets.append(None)
                    parents.append(current)
                    position_order.append(None)
                    rot_order.append(None)
                    channels.append(None)
                    current = len(names) - 1
                    continue

                if "}" in line:
                    if is_end_site:
                        is_end_site = False
                    else:
                        current = parents[current]
                    continue

                if is_end_site:
                    continue

                if "End Site" in line:
                    is_end_site = True
                    continue

                if "OFFSET" in line:
                    offsets[current] = [float(x) for x in line.split()[1:4]]
                    continue

                if "CHANNELS" in line:
                    words = line.split()
                    number_channels = int(words[1])
                    channels[current] = number_channels
                    if number_channels == 6:
                        position_order[current] = [
                            self.bvh_pos_map_num[x] for x in words[2 : 2 + 3]
                        ]
                        rot_order[current] = [
                            self.bvh_rot_map[x] for x in words[2 + 3 : 2 + 3 + 3]
                        ]
                    elif number_channels == 3:
                        rot_order[current] = [
                            self.bvh_rot_map[x] for x in words[2 : 2 + 3]
                        ]
                    else:
                        raise Exception("Unknown number of channels")
                    continue

                if "Frames" in line:
                    number_frames = int(line.split()[1])
                    offsets = np.array(offsets)
                    rot_order = np.array(rot_order)
                    positions = np.tile(offsets, (number_frames, 1)).reshape(
                        number_frames, len(offsets), 3
                    )
                    rotations = np.zeros((number_frames, len(names), 3))
                    continue

                if "Frame Time" in line:
                    frame_time = float(line.split()[2])
                    reading_frames = True
                    continue
            else:
                values = [float(x) for x in line.split()]
                i = 0
                for j in range(len(names)):
                    if channels[j] == 6:
                        positions[frame, j, position_order[j]] = values[i : i + 3]
                        rotations[frame, j] = values[i + 3 : i + 6]
                    elif channels[j] == 3:
                        rotations[frame, j] = values[i : i + 3]
                    i += channels[j]
                frame += 1

        f.close()

        self.data = {
            "names": names,
            "offsets": offsets,
            "parents": parents,
            "rot_order": rot_order,
            "positions": positions,
            "rotations": rotations,
            "frame_time": frame_time,
        }
        return self.data

    def save(self, filename, data):
        """
        Saves a BVH file from a dictionary with the data.

        Parameters
        ----------
        filename : str
            path to the file.
        data : dict
            dictionary with the data following the
            returned dict structure from load(...).
            Positions in data["positions"] are assumed to be X,Y,Z order.
            Rotations in data["rotations"] are assumed to be in the specified order in data["rot_order"].
        """

        with open(filename, "w") as f:
            tab = ""
            f.write("%sHIERARCHY\n" % tab)
            f.write("%sROOT %s\n" % (tab, data["names"][0]))
            f.write("%s{\n" % tab)
            tab += "\t"

            f.write(
                "%sOFFSET %f %f %f\n"
                % (
                    tab,
                    data["offsets"][0, 0],
                    data["offsets"][0, 1],
                    data["offsets"][0, 2],
                )
            )
            f.write(
                "%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n"
                % (
                    tab,
                    self.inv_bvh_rot_map[data["rot_order"][0, 0]],
                    self.inv_bvh_rot_map[data["rot_order"][0, 1]],
                    self.inv_bvh_rot_map[data["rot_order"][0, 2]],
                )
            )

            joint_order = [0]

            for i in range(len(data["parents"])):
                if data["parents"][i] == 0:
                    tab = self.save_joint(f, data, tab, i, joint_order)

            tab = tab[:-1]
            f.write("%s}\n" % tab)

            f.write("%sMOTION\n" % tab)
            f.write("%sFrames: %d\n" % (tab, data["positions"].shape[0]))
            f.write("%sFrame Time: %f\n" % (tab, data["frame_time"]))

            for i in range(data["positions"].shape[0]):
                for j in joint_order:
                    if j == 0:  # root
                        f.write(
                            "%f %f %f "
                            % (
                                data["positions"][i, j, 0],
                                data["positions"][i, j, 1],
                                data["positions"][i, j, 2],
                            )
                        )
                    f.write("%f %f %f " % tuple(data["rotations"][i, j]))
                f.write("\n")

    def save_joint(self, f, data, tab, i, joint_order):
        joint_order.append(i)

        f.write("%sJOINT %s\n" % (tab, data["names"][i]))
        f.write("%s{\n" % tab)
        tab += "\t"

        f.write(
            "%sOFFSET %f %f %f\n"
            % (
                tab,
                data["offsets"][i, 0],
                data["offsets"][i, 1],
                data["offsets"][i, 2],
            )
        )
        f.write(
            "%sCHANNELS 3 %s %s %s\n"
            % (
                tab,
                self.inv_bvh_rot_map[data["rot_order"][i, 0]],
                self.inv_bvh_rot_map[data["rot_order"][i, 1]],
                self.inv_bvh_rot_map[data["rot_order"][i, 2]],
            )
        )

        is_end_site = True

        for j in range(len(data["parents"])):
            if data["parents"][j] == i:
                tab = self.save_joint(f, data, tab, j, joint_order)
                is_end_site = False

        if is_end_site:
            f.write("%sEnd Site\n" % tab)
            f.write("%s{\n" % tab)
            tab += "\t"
            f.write("%sOFFSET %f %f %f\n" % (tab, 0.0, 0.0, 0.0))
            tab = tab[:-1]
            f.write("%s}\n" % tab)

        tab = tab[:-1]
        f.write("%s}\n" % tab)

        return tab
