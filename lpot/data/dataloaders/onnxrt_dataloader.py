#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from lpot.utils.utility import LazyImport
from .base_dataloader import BaseDataLoader
from .default_dataloader import DefaultDataLoader

class ONNXRTDataLoader(BaseDataLoader):
    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn,
                             sampler, batch_sampler, num_workers, pin_memory):
        return DefaultDataLoader(dataset, batch_size, last_batch, collate_fn,
                                 sampler, batch_sampler, num_workers, pin_memory)

